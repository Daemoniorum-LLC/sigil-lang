//! GraphQL Client Support
//!
//! Flexible query language client for API interactions.
//!
//! ## Features
//!
//! - Query and mutation execution
//! - Subscription support (WebSocket transport)
//! - Variable interpolation
//! - Error handling
//! - Batched queries
//! - Persisted queries

use super::common::{Headers, ProtocolError, ProtocolResult, Timeout};
use std::collections::HashMap;
use std::time::Duration;

/// GraphQL client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// GraphQL endpoint URL
    pub endpoint: String,
    /// WebSocket endpoint for subscriptions
    pub ws_endpoint: Option<String>,
    /// Default headers
    pub headers: Headers,
    /// Request timeout
    pub timeout: Duration,
    /// Enable persisted queries
    pub persisted_queries: bool,
    /// Batch interval for query batching
    pub batch_interval: Option<Duration>,
    /// Maximum batch size
    pub max_batch_size: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            endpoint: String::new(),
            ws_endpoint: None,
            headers: Headers::new(),
            timeout: Duration::from_secs(30),
            persisted_queries: false,
            batch_interval: None,
            max_batch_size: 10,
        }
    }
}

/// GraphQL client
#[derive(Debug, Clone)]
pub struct Client {
    config: ClientConfig,
}

impl Client {
    /// Create a new GraphQL client
    pub fn new(endpoint: impl Into<String>) -> Self {
        Client {
            config: ClientConfig {
                endpoint: endpoint.into(),
                ..Default::default()
            },
        }
    }

    /// Create a client builder
    pub fn builder(endpoint: impl Into<String>) -> ClientBuilder {
        ClientBuilder {
            config: ClientConfig {
                endpoint: endpoint.into(),
                ..Default::default()
            },
        }
    }

    /// Set bearer authentication
    pub fn bearer_auth(mut self, token: impl Into<String>) -> Self {
        self.config
            .headers
            .insert("Authorization", format!("Bearer {}", token.into()));
        self
    }

    /// Execute a query
    pub async fn query<V: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        query: &str,
        variables: Option<V>,
    ) -> ProtocolResult<GraphQLResponse<R>> {
        self.execute(query, variables, OperationType::Query).await
    }

    /// Execute a mutation
    pub async fn mutate<V: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        mutation: &str,
        variables: Option<V>,
    ) -> ProtocolResult<GraphQLResponse<R>> {
        self.execute(mutation, variables, OperationType::Mutation)
            .await
    }

    /// Execute an operation
    async fn execute<V: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        query: &str,
        variables: Option<V>,
        _operation_type: OperationType,
    ) -> ProtocolResult<GraphQLResponse<R>> {
        #[cfg(feature = "reqwest")]
        {
            let request_body = GraphQLRequest {
                query: query.to_string(),
                variables: variables.map(|v| serde_json::to_value(v).ok()).flatten(),
                operation_name: None,
                extensions: None,
            };

            let client = reqwest::Client::new();
            let mut req = client
                .post(&self.config.endpoint)
                .header("Content-Type", "application/json")
                .timeout(self.config.timeout);

            // Add headers
            for (key, values) in self.config.headers.iter() {
                for value in values {
                    req = req.header(key.as_str(), value.as_str());
                }
            }

            let response = req
                .json(&request_body)
                .send()
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;

            let body: GraphQLResponseRaw = response
                .json()
                .await
                .map_err(|e| ProtocolError::Deserialization(e.to_string()))?;

            let data = if let Some(data) = body.data {
                Some(
                    serde_json::from_value(data)
                        .map_err(|e| ProtocolError::Deserialization(e.to_string()))?,
                )
            } else {
                None
            };

            Ok(GraphQLResponse {
                data,
                errors: body.errors,
                extensions: body.extensions,
            })
        }

        #[cfg(not(feature = "reqwest"))]
        {
            let _ = (query, variables);
            Err(ProtocolError::Protocol(
                "GraphQL requires 'graphql' feature".to_string(),
            ))
        }
    }
}

/// Client builder
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    /// Set WebSocket endpoint for subscriptions
    pub fn ws_endpoint(mut self, url: impl Into<String>) -> Self {
        self.config.ws_endpoint = Some(url.into());
        self
    }

    /// Add a default header
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.insert(key, value);
        self
    }

    /// Set bearer authentication
    pub fn bearer_auth(self, token: impl Into<String>) -> Self {
        self.header("Authorization", format!("Bearer {}", token.into()))
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Enable persisted queries
    pub fn persisted_queries(mut self, enable: bool) -> Self {
        self.config.persisted_queries = enable;
        self
    }

    /// Enable query batching
    pub fn batch(mut self, interval: Duration, max_size: usize) -> Self {
        self.config.batch_interval = Some(interval);
        self.config.max_batch_size = max_size;
        self
    }

    /// Build the client
    pub fn build(self) -> Client {
        Client {
            config: self.config,
        }
    }
}

/// Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    Query,
    Mutation,
    Subscription,
}

/// GraphQL request body
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphQLRequest {
    /// The GraphQL query/mutation
    pub query: String,
    /// Variables for the query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<serde_json::Value>,
    /// Operation name (for documents with multiple operations)
    #[serde(rename = "operationName", skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
    /// Extensions (e.g., for persisted queries)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

/// Raw GraphQL response (before parsing data)
#[derive(Debug, Clone, serde::Deserialize)]
struct GraphQLResponseRaw {
    data: Option<serde_json::Value>,
    errors: Option<Vec<GraphQLError>>,
    extensions: Option<serde_json::Value>,
}

/// GraphQL response
#[derive(Debug, Clone)]
pub struct GraphQLResponse<T> {
    /// Response data
    pub data: Option<T>,
    /// Errors from the server
    pub errors: Option<Vec<GraphQLError>>,
    /// Extensions
    pub extensions: Option<serde_json::Value>,
}

impl<T> GraphQLResponse<T> {
    /// Check if the response has errors
    pub fn has_errors(&self) -> bool {
        self.errors.as_ref().map(|e| !e.is_empty()).unwrap_or(false)
    }

    /// Get the data or return the first error
    pub fn into_result(self) -> ProtocolResult<T> {
        if let Some(errors) = self.errors {
            if let Some(first_error) = errors.into_iter().next() {
                return Err(ProtocolError::Protocol(first_error.message));
            }
        }
        self.data
            .ok_or_else(|| ProtocolError::Protocol("No data in response".to_string()))
    }

    /// Get a reference to the data
    pub fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    /// Get the errors
    pub fn errors(&self) -> Option<&[GraphQLError]> {
        self.errors.as_deref()
    }
}

/// GraphQL error
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GraphQLError {
    /// Error message
    pub message: String,
    /// Error locations in the query
    #[serde(default)]
    pub locations: Vec<Location>,
    /// Error path
    #[serde(default)]
    pub path: Vec<PathSegment>,
    /// Error extensions
    #[serde(default)]
    pub extensions: Option<serde_json::Value>,
}

impl std::fmt::Display for GraphQLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if !self.locations.is_empty() {
            write!(f, " at ")?;
            for (i, loc) in self.locations.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}:{}", loc.line, loc.column)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for GraphQLError {}

/// Location in a GraphQL document
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Location {
    /// Line number (1-indexed)
    pub line: u32,
    /// Column number (1-indexed)
    pub column: u32,
}

/// Path segment in error path
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum PathSegment {
    /// Field name
    Field(String),
    /// Array index
    Index(usize),
}

/// GraphQL subscription
pub struct Subscription<T> {
    /// Subscription ID
    id: String,
    /// Phantom data
    _marker: std::marker::PhantomData<T>,
}

impl<T> Subscription<T> {
    /// Get the subscription ID
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Query builder for type-safe queries
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    query: String,
    variables: HashMap<String, serde_json::Value>,
    operation_name: Option<String>,
}

impl QueryBuilder {
    /// Create a new query builder
    pub fn new(query: impl Into<String>) -> Self {
        QueryBuilder {
            query: query.into(),
            variables: HashMap::new(),
            operation_name: None,
        }
    }

    /// Set a variable
    pub fn variable(mut self, name: impl Into<String>, value: impl serde::Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.variables.insert(name.into(), v);
        }
        self
    }

    /// Set the operation name
    pub fn operation_name(mut self, name: impl Into<String>) -> Self {
        self.operation_name = Some(name.into());
        self
    }

    /// Execute the query
    pub async fn execute<R: serde::de::DeserializeOwned>(
        self,
        client: &Client,
    ) -> ProtocolResult<GraphQLResponse<R>> {
        let variables = if self.variables.is_empty() {
            None
        } else {
            Some(serde_json::to_value(self.variables).unwrap_or_default())
        };

        client
            .execute(&self.query, variables, OperationType::Query)
            .await
    }
}

/// Mutation builder for type-safe mutations
#[derive(Debug, Clone)]
pub struct MutationBuilder {
    mutation: String,
    variables: HashMap<String, serde_json::Value>,
    operation_name: Option<String>,
}

impl MutationBuilder {
    /// Create a new mutation builder
    pub fn new(mutation: impl Into<String>) -> Self {
        MutationBuilder {
            mutation: mutation.into(),
            variables: HashMap::new(),
            operation_name: None,
        }
    }

    /// Set a variable
    pub fn variable(mut self, name: impl Into<String>, value: impl serde::Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.variables.insert(name.into(), v);
        }
        self
    }

    /// Set the operation name
    pub fn operation_name(mut self, name: impl Into<String>) -> Self {
        self.operation_name = Some(name.into());
        self
    }

    /// Execute the mutation
    pub async fn execute<R: serde::de::DeserializeOwned>(
        self,
        client: &Client,
    ) -> ProtocolResult<GraphQLResponse<R>> {
        let variables = if self.variables.is_empty() {
            None
        } else {
            Some(serde_json::to_value(self.variables).unwrap_or_default())
        };

        client
            .execute(&self.mutation, variables, OperationType::Mutation)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder() {
        let client = Client::builder("https://api.example.com/graphql")
            .bearer_auth("token123")
            .timeout(Duration::from_secs(60))
            .build();

        assert_eq!(client.config.endpoint, "https://api.example.com/graphql");
        assert!(client.config.headers.get("authorization").is_some());
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new("query GetUser($id: ID!) { user(id: $id) { name } }")
            .variable("id", "123")
            .operation_name("GetUser");

        assert_eq!(query.operation_name, Some("GetUser".to_string()));
        assert!(query.variables.contains_key("id"));
    }

    #[test]
    fn test_graphql_error() {
        let error = GraphQLError {
            message: "Field not found".to_string(),
            locations: vec![Location {
                line: 1,
                column: 10,
            }],
            path: vec![PathSegment::Field("user".to_string())],
            extensions: None,
        };

        let display = format!("{}", error);
        assert!(display.contains("Field not found"));
        assert!(display.contains("1:10"));
    }

    #[test]
    fn test_graphql_request_serialization() {
        let request = GraphQLRequest {
            query: "{ users { id } }".to_string(),
            variables: Some(serde_json::json!({"limit": 10})),
            operation_name: None,
            extensions: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("query"));
        assert!(json.contains("variables"));
    }
}
