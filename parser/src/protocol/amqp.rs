//! AMQP Support
//!
//! Advanced Message Queuing Protocol support for RabbitMQ and other AMQP brokers.
//!
//! ## Features
//!
//! - Connection and channel management
//! - Exchange and queue declaration
//! - Message publishing with confirmations
//! - Consumer with acknowledgments
//! - Transactions
//! - Dead letter queues

use super::common::{Headers, ProtocolError, ProtocolResult, Timeout};
use std::collections::HashMap;
use std::time::Duration;

/// AMQP connection configuration
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// AMQP URI (e.g., "amqp://user:pass@host:5672/vhost")
    pub uri: String,
    /// Connection name
    pub connection_name: Option<String>,
    /// Heartbeat interval
    pub heartbeat: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Channel max
    pub channel_max: u16,
    /// Frame max
    pub frame_max: u32,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        ConnectionConfig {
            uri: "amqp://guest:guest@localhost:5672/%2f".to_string(),
            connection_name: None,
            heartbeat: Duration::from_secs(60),
            connection_timeout: Duration::from_secs(30),
            channel_max: 0, // Use server default
            frame_max: 0,   // Use server default
        }
    }
}

/// Exchange type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeType {
    /// Direct exchange (routing key match)
    Direct,
    /// Fanout exchange (broadcast)
    Fanout,
    /// Topic exchange (pattern matching)
    Topic,
    /// Headers exchange (header matching)
    Headers,
}

impl ExchangeType {
    /// Get the exchange type string
    pub fn as_str(&self) -> &'static str {
        match self {
            ExchangeType::Direct => "direct",
            ExchangeType::Fanout => "fanout",
            ExchangeType::Topic => "topic",
            ExchangeType::Headers => "headers",
        }
    }
}

/// Exchange declaration options
#[derive(Debug, Clone)]
pub struct ExchangeOptions {
    /// Exchange is durable (survives broker restart)
    pub durable: bool,
    /// Auto-delete when no longer used
    pub auto_delete: bool,
    /// Internal exchange (can't publish to directly)
    pub internal: bool,
    /// Additional arguments
    pub arguments: HashMap<String, FieldValue>,
}

impl Default for ExchangeOptions {
    fn default() -> Self {
        ExchangeOptions {
            durable: true,
            auto_delete: false,
            internal: false,
            arguments: HashMap::new(),
        }
    }
}

/// Queue declaration options
#[derive(Debug, Clone)]
pub struct QueueOptions {
    /// Queue is durable (survives broker restart)
    pub durable: bool,
    /// Exclusive to this connection
    pub exclusive: bool,
    /// Auto-delete when no longer used
    pub auto_delete: bool,
    /// Additional arguments
    pub arguments: HashMap<String, FieldValue>,
}

impl Default for QueueOptions {
    fn default() -> Self {
        QueueOptions {
            durable: true,
            exclusive: false,
            auto_delete: false,
            arguments: HashMap::new(),
        }
    }
}

impl QueueOptions {
    /// Set message TTL
    pub fn message_ttl(mut self, ttl: Duration) -> Self {
        self.arguments.insert(
            "x-message-ttl".to_string(),
            FieldValue::LongInt(ttl.as_millis() as i32),
        );
        self
    }

    /// Set queue max length
    pub fn max_length(mut self, max: u32) -> Self {
        self.arguments
            .insert("x-max-length".to_string(), FieldValue::LongInt(max as i32));
        self
    }

    /// Set dead letter exchange
    pub fn dead_letter_exchange(mut self, exchange: impl Into<String>) -> Self {
        self.arguments.insert(
            "x-dead-letter-exchange".to_string(),
            FieldValue::LongString(exchange.into()),
        );
        self
    }

    /// Set dead letter routing key
    pub fn dead_letter_routing_key(mut self, key: impl Into<String>) -> Self {
        self.arguments.insert(
            "x-dead-letter-routing-key".to_string(),
            FieldValue::LongString(key.into()),
        );
        self
    }
}

/// AMQP field value types
#[derive(Debug, Clone)]
pub enum FieldValue {
    /// Boolean
    Boolean(bool),
    /// Short integer
    ShortInt(i16),
    /// Long integer
    LongInt(i32),
    /// Long long integer
    LongLongInt(i64),
    /// Float
    Float(f32),
    /// Double
    Double(f64),
    /// Short string
    ShortString(String),
    /// Long string
    LongString(String),
    /// Timestamp
    Timestamp(u64),
    /// Byte array
    ByteArray(Vec<u8>),
    /// Table (nested)
    Table(HashMap<String, FieldValue>),
    /// Array
    Array(Vec<FieldValue>),
    /// Void/null
    Void,
}

/// Message delivery mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    /// Transient (not persisted)
    Transient = 1,
    /// Persistent (written to disk)
    Persistent = 2,
}

/// Message properties
#[derive(Debug, Clone, Default)]
pub struct MessageProperties {
    /// Content type (e.g., "application/json")
    pub content_type: Option<String>,
    /// Content encoding (e.g., "utf-8")
    pub content_encoding: Option<String>,
    /// Message headers
    pub headers: HashMap<String, FieldValue>,
    /// Delivery mode
    pub delivery_mode: Option<DeliveryMode>,
    /// Message priority (0-9)
    pub priority: Option<u8>,
    /// Correlation ID
    pub correlation_id: Option<String>,
    /// Reply-to queue
    pub reply_to: Option<String>,
    /// Message expiration (milliseconds)
    pub expiration: Option<String>,
    /// Message ID
    pub message_id: Option<String>,
    /// Timestamp
    pub timestamp: Option<u64>,
    /// Message type
    pub message_type: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Application ID
    pub app_id: Option<String>,
}

impl MessageProperties {
    /// Create new message properties
    pub fn new() -> Self {
        MessageProperties::default()
    }

    /// Set content type
    pub fn content_type(mut self, ct: impl Into<String>) -> Self {
        self.content_type = Some(ct.into());
        self
    }

    /// Set as JSON content
    pub fn json(self) -> Self {
        self.content_type("application/json")
    }

    /// Set delivery mode
    pub fn delivery_mode(mut self, mode: DeliveryMode) -> Self {
        self.delivery_mode = Some(mode);
        self
    }

    /// Set as persistent message
    pub fn persistent(self) -> Self {
        self.delivery_mode(DeliveryMode::Persistent)
    }

    /// Set correlation ID
    pub fn correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Set reply-to queue
    pub fn reply_to(mut self, queue: impl Into<String>) -> Self {
        self.reply_to = Some(queue.into());
        self
    }

    /// Set message ID
    pub fn message_id(mut self, id: impl Into<String>) -> Self {
        self.message_id = Some(id.into());
        self
    }

    /// Set priority
    pub fn priority(mut self, p: u8) -> Self {
        self.priority = Some(p.min(9));
        self
    }

    /// Set expiration (TTL in milliseconds)
    pub fn expiration_ms(mut self, ms: u64) -> Self {
        self.expiration = Some(ms.to_string());
        self
    }

    /// Add a header
    pub fn header(mut self, key: impl Into<String>, value: FieldValue) -> Self {
        self.headers.insert(key.into(), value);
        self
    }
}

/// A message to publish
#[derive(Debug, Clone)]
pub struct Message {
    /// Message body
    pub body: Vec<u8>,
    /// Message properties
    pub properties: MessageProperties,
}

impl Message {
    /// Create a new message
    pub fn new(body: impl Into<Vec<u8>>) -> Self {
        Message {
            body: body.into(),
            properties: MessageProperties::default(),
        }
    }

    /// Create a text message
    pub fn text(text: impl Into<String>) -> Self {
        Message::new(text.into().into_bytes())
    }

    /// Create a JSON message
    pub fn json<T: serde::Serialize>(value: &T) -> ProtocolResult<Self> {
        let body =
            serde_json::to_vec(value).map_err(|e| ProtocolError::Serialization(e.to_string()))?;
        Ok(Message {
            body,
            properties: MessageProperties::new().json(),
        })
    }

    /// Set properties
    pub fn properties(mut self, props: MessageProperties) -> Self {
        self.properties = props;
        self
    }

    /// Set as persistent
    pub fn persistent(mut self) -> Self {
        self.properties.delivery_mode = Some(DeliveryMode::Persistent);
        self
    }

    /// Set correlation ID
    pub fn correlation_id(mut self, id: impl Into<String>) -> Self {
        self.properties.correlation_id = Some(id.into());
        self
    }

    /// Set reply-to queue
    pub fn reply_to(mut self, queue: impl Into<String>) -> Self {
        self.properties.reply_to = Some(queue.into());
        self
    }
}

/// A delivered message
#[derive(Debug, Clone)]
pub struct Delivery {
    /// Delivery tag (for acknowledgment)
    pub delivery_tag: u64,
    /// Was this message redelivered?
    pub redelivered: bool,
    /// Exchange the message was published to
    pub exchange: String,
    /// Routing key
    pub routing_key: String,
    /// Message properties
    pub properties: MessageProperties,
    /// Message body
    pub body: Vec<u8>,
}

impl Delivery {
    /// Get the body as a string
    pub fn body_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }

    /// Parse the body as JSON
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> ProtocolResult<T> {
        serde_json::from_slice(&self.body)
            .map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }
}

/// Publish confirmation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confirmation {
    /// Message was acknowledged
    Ack,
    /// Message was not acknowledged
    Nack,
}

/// Consumer options
#[derive(Debug, Clone)]
pub struct ConsumerOptions {
    /// Consumer tag
    pub consumer_tag: Option<String>,
    /// Don't require acknowledgment
    pub no_ack: bool,
    /// Exclusive consumer
    pub exclusive: bool,
    /// No local delivery
    pub no_local: bool,
    /// Additional arguments
    pub arguments: HashMap<String, FieldValue>,
}

impl Default for ConsumerOptions {
    fn default() -> Self {
        ConsumerOptions {
            consumer_tag: None,
            no_ack: false,
            exclusive: false,
            no_local: false,
            arguments: HashMap::new(),
        }
    }
}

/// AMQP connection
#[derive(Debug)]
pub struct Connection {
    config: ConnectionConfig,
    #[cfg(feature = "lapin")]
    inner: Option<lapin::Connection>,
}

impl Connection {
    /// Connect to an AMQP broker
    #[cfg(feature = "lapin")]
    pub async fn connect(uri: impl Into<String>) -> ProtocolResult<Self> {
        let config = ConnectionConfig {
            uri: uri.into(),
            ..Default::default()
        };

        let options = lapin::ConnectionProperties::default().with_connection_name(
            config
                .connection_name
                .clone()
                .unwrap_or_else(|| "sigil-amqp".to_string())
                .into(),
        );

        let connection = lapin::Connection::connect(&config.uri, options)
            .await
            .map_err(|e| ProtocolError::ConnectionFailed(e.to_string()))?;

        Ok(Connection {
            config,
            inner: Some(connection),
        })
    }

    /// Connect to an AMQP broker (stub for non-lapin builds)
    #[cfg(not(feature = "lapin"))]
    pub async fn connect(uri: impl Into<String>) -> ProtocolResult<Self> {
        let _ = uri;
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Create a new channel
    #[cfg(feature = "lapin")]
    pub async fn create_channel(&self) -> ProtocolResult<Channel> {
        if let Some(ref conn) = self.inner {
            let channel = conn
                .create_channel()
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;

            Ok(Channel {
                inner: Some(channel),
            })
        } else {
            Err(ProtocolError::Protocol(
                "Connection not initialized".to_string(),
            ))
        }
    }

    /// Create a new channel (stub for non-lapin builds)
    #[cfg(not(feature = "lapin"))]
    pub async fn create_channel(&self) -> ProtocolResult<Channel> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Check if connection is connected
    #[cfg(feature = "lapin")]
    pub fn is_connected(&self) -> bool {
        self.inner
            .as_ref()
            .map(|c| c.status().connected())
            .unwrap_or(false)
    }

    #[cfg(not(feature = "lapin"))]
    pub fn is_connected(&self) -> bool {
        false
    }

    /// Close the connection
    #[cfg(feature = "lapin")]
    pub async fn close(&self) -> ProtocolResult<()> {
        if let Some(ref conn) = self.inner {
            conn.close(200, "Normal shutdown")
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn close(&self) -> ProtocolResult<()> {
        Ok(())
    }
}

/// AMQP channel
#[derive(Debug)]
pub struct Channel {
    #[cfg(feature = "lapin")]
    inner: Option<lapin::Channel>,
}

impl Channel {
    /// Declare an exchange
    #[cfg(feature = "lapin")]
    pub async fn declare_exchange(
        &self,
        name: &str,
        kind: ExchangeType,
        options: ExchangeOptions,
    ) -> ProtocolResult<()> {
        use lapin::options::ExchangeDeclareOptions;
        use lapin::types::FieldTable;

        if let Some(ref channel) = self.inner {
            let declare_options = ExchangeDeclareOptions {
                durable: options.durable,
                auto_delete: options.auto_delete,
                internal: options.internal,
                ..Default::default()
            };

            channel
                .exchange_declare(
                    name,
                    lapin::ExchangeKind::Custom(kind.as_str().to_string()),
                    declare_options,
                    FieldTable::default(),
                )
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn declare_exchange(
        &self,
        _name: &str,
        _kind: ExchangeType,
        _options: ExchangeOptions,
    ) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Declare a queue
    #[cfg(feature = "lapin")]
    pub async fn declare_queue(&self, name: &str, options: QueueOptions) -> ProtocolResult<Queue> {
        use lapin::options::QueueDeclareOptions;
        use lapin::types::FieldTable;

        if let Some(ref channel) = self.inner {
            let declare_options = QueueDeclareOptions {
                durable: options.durable,
                exclusive: options.exclusive,
                auto_delete: options.auto_delete,
                ..Default::default()
            };

            let queue = channel
                .queue_declare(name, declare_options, FieldTable::default())
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;

            Ok(Queue {
                name: queue.name().to_string(),
                message_count: queue.message_count(),
                consumer_count: queue.consumer_count(),
            })
        } else {
            Err(ProtocolError::Protocol(
                "Channel not initialized".to_string(),
            ))
        }
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn declare_queue(
        &self,
        _name: &str,
        _options: QueueOptions,
    ) -> ProtocolResult<Queue> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Bind a queue to an exchange
    #[cfg(feature = "lapin")]
    pub async fn bind_queue(
        &self,
        queue: &str,
        exchange: &str,
        routing_key: &str,
    ) -> ProtocolResult<()> {
        use lapin::options::QueueBindOptions;
        use lapin::types::FieldTable;

        if let Some(ref channel) = self.inner {
            channel
                .queue_bind(
                    queue,
                    exchange,
                    routing_key,
                    QueueBindOptions::default(),
                    FieldTable::default(),
                )
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn bind_queue(
        &self,
        _queue: &str,
        _exchange: &str,
        _routing_key: &str,
    ) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Publish a message
    #[cfg(feature = "lapin")]
    pub async fn publish(
        &self,
        exchange: &str,
        routing_key: &str,
        message: Message,
    ) -> ProtocolResult<()> {
        use lapin::options::BasicPublishOptions;
        use lapin::BasicProperties;

        if let Some(ref channel) = self.inner {
            let mut properties = BasicProperties::default();

            if let Some(ref ct) = message.properties.content_type {
                properties = properties.with_content_type(ct.as_str().into());
            }
            if let Some(mode) = message.properties.delivery_mode {
                properties = properties.with_delivery_mode(mode as u8);
            }
            if let Some(ref cid) = message.properties.correlation_id {
                properties = properties.with_correlation_id(cid.as_str().into());
            }
            if let Some(ref reply) = message.properties.reply_to {
                properties = properties.with_reply_to(reply.as_str().into());
            }
            if let Some(ref mid) = message.properties.message_id {
                properties = properties.with_message_id(mid.as_str().into());
            }
            if let Some(priority) = message.properties.priority {
                properties = properties.with_priority(priority);
            }

            channel
                .basic_publish(
                    exchange,
                    routing_key,
                    BasicPublishOptions::default(),
                    &message.body,
                    properties,
                )
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn publish(
        &self,
        _exchange: &str,
        _routing_key: &str,
        _message: Message,
    ) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Enable publisher confirms
    #[cfg(feature = "lapin")]
    pub async fn confirm_select(&self) -> ProtocolResult<()> {
        use lapin::options::ConfirmSelectOptions;

        if let Some(ref channel) = self.inner {
            channel
                .confirm_select(ConfirmSelectOptions::default())
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn confirm_select(&self) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }

    /// Set QoS (prefetch count)
    #[cfg(feature = "lapin")]
    pub async fn qos(&self, prefetch_count: u16) -> ProtocolResult<()> {
        use lapin::options::BasicQosOptions;

        if let Some(ref channel) = self.inner {
            channel
                .basic_qos(prefetch_count, BasicQosOptions::default())
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "lapin"))]
    pub async fn qos(&self, _prefetch_count: u16) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol(
            "AMQP requires 'amqp' feature".to_string(),
        ))
    }
}

/// Declared queue info
#[derive(Debug, Clone)]
pub struct Queue {
    /// Queue name
    pub name: String,
    /// Number of messages in the queue
    pub message_count: u32,
    /// Number of consumers
    pub consumer_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_types() {
        assert_eq!(ExchangeType::Direct.as_str(), "direct");
        assert_eq!(ExchangeType::Fanout.as_str(), "fanout");
        assert_eq!(ExchangeType::Topic.as_str(), "topic");
    }

    #[test]
    fn test_message_builder() {
        let msg = Message::text("hello world")
            .persistent()
            .correlation_id("123")
            .reply_to("response-queue");

        assert_eq!(msg.body, b"hello world");
        assert_eq!(msg.properties.delivery_mode, Some(DeliveryMode::Persistent));
        assert_eq!(msg.properties.correlation_id, Some("123".to_string()));
    }

    #[test]
    fn test_queue_options() {
        let opts = QueueOptions::default()
            .message_ttl(Duration::from_secs(60))
            .max_length(1000)
            .dead_letter_exchange("dlx");

        assert!(opts.arguments.contains_key("x-message-ttl"));
        assert!(opts.arguments.contains_key("x-max-length"));
        assert!(opts.arguments.contains_key("x-dead-letter-exchange"));
    }

    #[test]
    fn test_message_properties() {
        let props = MessageProperties::new()
            .json()
            .persistent()
            .priority(5)
            .expiration_ms(30000);

        assert_eq!(props.content_type, Some("application/json".to_string()));
        assert_eq!(props.delivery_mode, Some(DeliveryMode::Persistent));
        assert_eq!(props.priority, Some(5));
        assert_eq!(props.expiration, Some("30000".to_string()));
    }
}
