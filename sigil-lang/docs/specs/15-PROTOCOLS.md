# Sigil Protocol Specification

> *"Communication across boundaries requires understanding of many tongues."*

## 1. Philosophy: Universal Communication

Sigil's protocol support embodies the principle of universal communication—enabling programs to speak
many "languages" (protocols) fluently while maintaining type safety and evidentiality tracking.

| Protocol | Cultural Metaphor | Use Case |
|----------|------------------|----------|
| **gRPC** | Structured Discourse | High-performance service-to-service |
| **HTTP** | Universal Tongue | Web APIs, REST services |
| **WebSocket** | Continuous Dialogue | Real-time bidirectional communication |
| **GraphQL** | Precise Inquiry | Flexible data queries |
| **Kafka** | Ancestral Memory | Event streaming, message persistence |
| **AMQP** | Market Exchange | Message queuing, pub/sub |

---

## 2. Module Structure

```
std::protocol
├── grpc           # gRPC client and server
│   ├── Channel
│   ├── Client
│   ├── Server
│   ├── Service
│   └── streaming
│
├── http           # HTTP/1.1 and HTTP/2
│   ├── Client
│   ├── Request
│   ├── Response
│   ├── Method
│   └── Headers
│
├── ws             # WebSocket
│   ├── Connection
│   ├── Message
│   └── Frame
│
├── graphql        # GraphQL client
│   ├── Client
│   ├── Query
│   ├── Mutation
│   └── Subscription
│
├── kafka          # Apache Kafka
│   ├── Producer
│   ├── Consumer
│   ├── Topic
│   └── Message
│
├── amqp           # AMQP/RabbitMQ
│   ├── Connection
│   ├── Channel
│   ├── Queue
│   └── Exchange
│
└── common         # Shared abstractions
    ├── Uri
    ├── Headers
    ├── Status
    └── Timeout
```

---

## 3. gRPC Support

### 3.1 Client Usage

```sigil
use protocol::grpc::{Channel, Client}

// Create a channel to the server
let channel = Channel::connect("http://localhost:50051")|await~

// Create a client from proto definition
let client = GreeterClient::new(channel)

// Unary call
let response~ = client|say_hello(HelloRequest { name: "Sigil" })|await

// Server streaming
let stream~ = client|list_features(Rectangle { ... })|await
stream~|for_each·flow{feature =>
    print!("Feature: {feature.name}")
}

// Client streaming
let (sink, response) = client|record_route()|await
locations|for_each{loc => sink|send(loc)}
sink|close
let summary~ = response|await

// Bidirectional streaming
let stream~ = client|route_chat()|await
stream~|for_each_concurrent{msg =>
    // Handle incoming while sending outgoing
    stream~|send(response_for(msg))
}
```

### 3.2 Server Definition

```sigil
use protocol::grpc::{Server, Service, Request, Response, Status}

// Define service implementation
service Greeter {
    async fn say_hello(request: Request<HelloRequest>) -> Result<Response<HelloReply>!, Status~> {
        let name = request|into_inner.name
        Ok(Response::new(HelloReply {
            message: "Hello, {name}!"
        }))
    }

    async fn list_features(request: Request<Rectangle>) -> impl Stream<Feature~> {
        let rect = request|into_inner
        features_in_rect(rect)|stream
    }
}

// Start server
async fn main() {
    let addr = "0.0.0.0:50051"|parse!

    Server::builder()
        |add_service(GreeterServer::new(GreeterImpl))
        |serve(addr)
        |await
}
```

### 3.3 Proto Integration

```sigil
// Proto files are compiled at build time
//@ proto: "protos/greeter.proto"

// Or inline proto definition
proto! {
    syntax = "proto3";

    message HelloRequest {
        string name = 1;
    }

    message HelloReply {
        string message = 1;
    }

    service Greeter {
        rpc SayHello (HelloRequest) returns (HelloReply);
    }
}
```

---

## 4. HTTP Support

### 4.1 HTTP Client

```sigil
use protocol::http::{Client, Request, Method, Headers}

// Simple GET
let response~ = http::get("https://api.example.com/users")|await

// With client instance
let client = Client::new()
    |timeout(30·sec)
    |default_headers([("Authorization", "Bearer {token}")])

// GET with query params
let users~ = client
    |get("https://api.example.com/users")
    |query([("page", "1"), ("limit", "10")])
    |send|await
    |json::<[User]>|await

// POST with JSON body
let created~ = client
    |post("https://api.example.com/users")
    |json(User { name: "Alice", email: "alice@example.com" })
    |send|await
    |json::<User>|await

// Multipart form
let uploaded~ = client
    |post("https://api.example.com/upload")
    |multipart([
        ("file", File::open("document.pdf")|await~),
        ("description", "My document")
    ])
    |send|await

// Streaming response
let stream~ = client
    |get("https://api.example.com/large-file")
    |send|await
    |bytes_stream

stream~|for_each_chunk{chunk =>
    file|write(chunk)|await
}
```

### 4.2 HTTP Response Handling

```sigil
// Response structure
struct Response {
    status: StatusCode,
    headers: Headers,
    body: Body,
}

impl Response {
    fn status(&self) -> StatusCode
    fn headers(&self) -> &Headers
    fn text(self) -> impl Future<String~>
    fn json<T: Deserialize>(self) -> impl Future<T~>
    fn bytes(self) -> impl Future<[u8]~>
    fn bytes_stream(self) -> impl Stream<[u8]~>
}

// Status handling
let response~ = client|get(url)|send|await

match response~.status {
    StatusCode::OK => process(response~|json|await),
    StatusCode::NOT_FOUND => Err(NotFoundError),
    StatusCode::UNAUTHORIZED => refresh_and_retry(),
    status if status.is_server_error() => retry_with_backoff(),
    _ => Err(UnexpectedStatus(status)),
}
```

### 4.3 Request Builder

```sigil
// Fluent request building
let request = Request::builder()
    |method(Method::POST)
    |uri("https://api.example.com/data")
    |header("Content-Type", "application/json")
    |header("X-Custom-Header", "value")
    |timeout(10·sec)
    |body(json_body)
    |build

let response~ = client|execute(request)|await
```

---

## 5. WebSocket Support

### 5.1 WebSocket Client

```sigil
use protocol::ws::{Connection, Message}

// Connect to WebSocket server
let (ws, response~) = ws::connect("wss://echo.example.com/socket")|await

// Send messages
ws|send(Message::Text("Hello, WebSocket!"))|await
ws|send(Message::Binary(bytes))|await

// Receive messages
loop {
    match ws|recv|await {
        Some(Message::Text(text)) => handle_text(text),
        Some(Message::Binary(data)) => handle_binary(data),
        Some(Message::Ping(data)) => ws|send(Message::Pong(data))|await,
        Some(Message::Close(_)) => break,
        None => break,
    }
}

// Close connection
ws|close(CloseCode::Normal, "Goodbye")|await
```

### 5.2 WebSocket Streams

```sigil
// Treat WebSocket as stream
let ws~ = ws::connect(url)|await

// Split into sender and receiver
let (sink, stream) = ws~|split

// Process incoming as stream
stream
    |filter_map{msg => match msg {
        Message::Text(t) => Some(t|json::<Event>),
        _ => None,
    }}
    |for_each·flow{event =>
        handle_event(event)|await
    }

// Send from another task
sink|send_all(outgoing_messages)|await
```

### 5.3 Reconnecting WebSocket

```sigil
use protocol::ws::ReconnectingConnection

// Auto-reconnecting WebSocket
let ws = ReconnectingConnection::new(url)
    |reconnect_delay(1·sec, max: 30·sec)  // Exponential backoff
    |max_reconnects(10)
    |on_reconnect{|| print!("Reconnected!")}

// Use like normal WebSocket
ws|send(message)|await
```

---

## 6. GraphQL Support

### 6.1 GraphQL Client

```sigil
use protocol::graphql::{Client, Query, Mutation}

let client = graphql::Client::new("https://api.example.com/graphql")
    |auth_bearer(token)

// Query
let users~ = client|query(graphql! {
    query GetUsers($first: Int!) {
        users(first: $first) {
            id
            name
            email
        }
    }
}, { first: 10 })|await

// Mutation
let user~ = client|mutate(graphql! {
    mutation CreateUser($input: CreateUserInput!) {
        createUser(input: $input) {
            id
            name
        }
    }
}, { input: { name: "Alice", email: "alice@example.com" } })|await

// Subscription
let stream~ = client|subscribe(graphql! {
    subscription OnUserCreated {
        userCreated {
            id
            name
        }
    }
})|await

stream~|for_each·flow{user =>
    print!("New user: {user.name}")
}
```

### 6.2 Type-Safe Queries

```sigil
// Generate types from schema
//@ graphql_schema: "schema.graphql"

// Typed query
struct GetUsersQuery;

impl GraphQLQuery for GetUsersQuery {
    type Variables = GetUsersVariables
    type Response = GetUsersResponse

    const QUERY: &str = "query GetUsers($first: Int!) { ... }"
}

let response~ = client|query::<GetUsersQuery>({ first: 10 })|await
let users: [User] = response~.data.users
```

---

## 7. Kafka Support

### 7.1 Kafka Producer

```sigil
use protocol::kafka::{Producer, ProducerConfig, Record}

let producer = Producer::new(ProducerConfig {
    bootstrap_servers: ["localhost:9092"],
    acks: Acks::All,
    compression: Compression::Snappy,
})

// Send single message
producer|send(Record {
    topic: "events",
    key: Some("user-123"),
    value: event|json::to_bytes,
    headers: [("source", "sigil-app")],
})|await

// Send batch
let records = events|τ{e => Record::new("events", e|json::to_bytes)}
producer|send_batch(records)|await

// Transactional producer
producer|begin_transaction
try {
    producer|send(record1)|await
    producer|send(record2)|await
    producer|commit_transaction|await
} catch {
    producer|abort_transaction|await
}
```

### 7.2 Kafka Consumer

```sigil
use protocol::kafka::{Consumer, ConsumerConfig, Offset}

let consumer = Consumer::new(ConsumerConfig {
    bootstrap_servers: ["localhost:9092"],
    group_id: "my-consumer-group",
    auto_offset_reset: Offset::Earliest,
})

// Subscribe to topics
consumer|subscribe(["events", "notifications"])

// Poll for messages
loop {
    let batch~ = consumer|poll(100·ms)|await
    for record in batch~ {
        let event: Event = record.value|json::from_bytes~
        process_event(event)|await
        consumer|commit(record)|await
    }
}

// Stream interface
consumer
    |stream
    |for_each·flow{record =>
        let event: Event~ = record.value|json::from_bytes
        process_event(event~)|await
        consumer|commit(record)|await
    }
```

### 7.3 Consumer Groups

```sigil
// Partition assignment callback
consumer|on_partitions_assigned{partitions =>
    print!("Assigned: {partitions}")
    // Load offsets from external storage if needed
}

consumer|on_partitions_revoked{partitions =>
    print!("Revoked: {partitions}")
    // Commit offsets before rebalance
    consumer|commit_sync|await
}
```

---

## 8. AMQP Support

### 8.1 AMQP Connection

```sigil
use protocol::amqp::{Connection, Channel, Queue, Exchange}

let conn = amqp::connect("amqp://guest:guest@localhost:5672")|await~

let channel = conn|create_channel|await~

// Declare exchange
channel|declare_exchange("events", ExchangeType::Topic, {
    durable: true,
    auto_delete: false,
})|await

// Declare queue
let queue = channel|declare_queue("my-queue", {
    durable: true,
    exclusive: false,
})|await

// Bind queue to exchange
channel|bind_queue(queue.name, "events", "user.*")|await
```

### 8.2 Publishing

```sigil
// Publish message
channel|publish("events", "user.created", message, {
    content_type: "application/json",
    delivery_mode: DeliveryMode::Persistent,
    headers: [("version", "1.0")],
})|await

// With confirmation
channel|enable_publisher_confirms|await
let confirm = channel|publish_with_confirm(exchange, key, message)|await
match confirm {
    Confirm::Ack => print!("Message confirmed"),
    Confirm::Nack => retry_publish(),
}
```

### 8.3 Consuming

```sigil
// Basic consumer
channel
    |consume("my-queue", { no_ack: false })
    |for_each·flow{delivery =>
        let message: Event = delivery.body|json::from_bytes~
        process(message)|await
        delivery|ack|await
    }

// With manual acknowledgment
channel
    |consume("my-queue")
    |for_each·flow{delivery =>
        match process(delivery.body) {
            Ok(_) => delivery|ack|await,
            Err(e) if e.is_retryable => delivery|nack(requeue: true)|await,
            Err(_) => delivery|reject|await,
        }
    }
```

---

## 9. Common Abstractions

### 9.1 URI and Headers

```sigil
use protocol::common::{Uri, Headers}

// Parse URI
let uri: Uri = "https://user:pass@example.com:8080/path?query=value"|parse!

uri.scheme      // "https"
uri.host        // "example.com"
uri.port        // Some(8080)
uri.path        // "/path"
uri.query       // Some("query=value")
uri.userinfo    // Some("user:pass")

// Headers
let headers = Headers::new()
    |insert("Content-Type", "application/json")
    |insert("Authorization", "Bearer {token}")

let content_type? = headers|get("Content-Type")
```

### 9.2 Status Codes

```sigil
use protocol::common::StatusCode

// HTTP-style status codes used across protocols
StatusCode::OK                  // 200
StatusCode::CREATED             // 201
StatusCode::BAD_REQUEST         // 400
StatusCode::UNAUTHORIZED        // 401
StatusCode::NOT_FOUND           // 404
StatusCode::INTERNAL_ERROR      // 500

// gRPC status codes
use protocol::grpc::Status

Status::OK
Status::CANCELLED
Status::INVALID_ARGUMENT
Status::NOT_FOUND
Status::ALREADY_EXISTS
Status::PERMISSION_DENIED
Status::RESOURCE_EXHAUSTED
Status::INTERNAL
Status::UNAVAILABLE
```

### 9.3 Timeouts and Retries

```sigil
use protocol::common::{Timeout, Retry, Backoff}

// Timeout configuration
let client = http::Client::new()
    |connect_timeout(5·sec)
    |read_timeout(30·sec)
    |total_timeout(60·sec)

// Retry policy
let client = client
    |retry(Retry::new()
        |max_attempts(3)
        |backoff(Backoff::Exponential {
            initial: 100·ms,
            factor: 2.0,
            max: 10·sec,
        })
        |retry_on(|status| status.is_server_error())
    )

// Circuit breaker
let client = client
    |circuit_breaker(CircuitBreaker::new()
        |failure_threshold(5)
        |success_threshold(2)
        |timeout(30·sec)
    )
```

---

## 10. Evidentiality in Protocols

### 10.1 Network Evidence

```sigil
// All network responses are "reported" (~) by default
let response~ = http::get(url)|await

// The data came from an external source
let user~ = response~|json::<User>|await

// Validate to promote to known (!)
let validated_user! = user~|validate!{u =>
    u.email|is_valid_email && u.age >= 0
}

// Or use type system
struct ValidatedUser {
    email: Email!,  // Known valid email
    age: u8!,       // Known non-negative
}
```

### 10.2 Error Evidence

```sigil
// Network errors are uncertain (?)
let result? = http::get(url)|await|catch

match result? {
    Ok(response~) => process(response~),
    Err(NetworkError::Timeout) => retry(),
    Err(NetworkError::ConnectionRefused) => use_fallback(),
    Err(e) => Err(e)~,  // Propagate as reported error
}

// Retry changes evidence
let response! = http::get(url)
    |retry(3)
    |await!  // Panic on final failure, or return known value
```

---

## 11. Protocol Morpheme Summary

| Morpheme | Meaning | Example |
|----------|---------|---------|
| `·get` | HTTP GET | `http·get(url)` |
| `·post` | HTTP POST | `http·post(url)` |
| `·send` | Send message | `ws·send(msg)` |
| `·recv` | Receive message | `ws·recv` |
| `·query` | GraphQL query | `client·query(q)` |
| `·mutate` | GraphQL mutation | `client·mutate(m)` |
| `·subscribe` | Subscribe to stream | `client·subscribe(s)` |
| `·publish` | Publish message | `kafka·publish(msg)` |
| `·consume` | Consume messages | `kafka·consume(topic)` |
| `·connect` | Establish connection | `grpc·connect(addr)` |
| `·stream` | Convert to stream | `response·stream` |
| `·json` | Parse/serialize JSON | `body·json::<T>` |

---

## 12. Complete Example: Microservice

```sigil
//! Order processing microservice with gRPC, Kafka, and HTTP

use protocol::{grpc, http, kafka}
use concurrency::{scope, flow}

// gRPC service definition
service OrderService {
    async fn create_order(req: Request<CreateOrderRequest>) -> Result<Response<Order>!, Status~> {
        let order_req = req|into_inner

        // Validate via external HTTP service
        let customer~ = http::get("http://customers/api/v1/{order_req.customer_id}")
            |await~
            |json::<Customer>
            |await~

        if !customer~.active {
            return Err(Status::permission_denied("Customer inactive"))
        }

        // Create order
        let order = Order::create(order_req, customer~)|await!

        // Publish event to Kafka
        kafka_producer|send(Record {
            topic: "order-events",
            key: order.id|to_string,
            value: OrderCreatedEvent { order: order.clone() }|json::to_bytes,
        })|await

        Ok(Response::new(order))
    }

    async fn stream_order_updates(req: Request<OrderId>) -> impl Stream<OrderUpdate~> {
        let order_id = req|into_inner.id

        // Subscribe to Kafka topic and filter for this order
        kafka_consumer
            |stream("order-updates")
            |filter·flow{record =>
                record.key == order_id|to_string
            }
            |map·flow{record =>
                record.value|json::<OrderUpdate>~
            }
    }
}

async fn main() {
    // Initialize clients
    let kafka_producer = kafka::Producer::new(kafka_config)|await
    let kafka_consumer = kafka::Consumer::new(kafka_config)|await

    // Start gRPC server
    let addr = "0.0.0.0:50051"|parse!

    print!("Starting OrderService on {addr}")

    grpc::Server::builder()
        |add_service(OrderServiceServer::new(OrderServiceImpl {
            kafka_producer,
            kafka_consumer,
        }))
        |serve(addr)
        |await
}
```

---

## 13. Security Considerations

### 13.1 TLS/mTLS

```sigil
use protocol::tls::{TlsConfig, ClientAuth}

// HTTP with TLS
let client = http::Client::new()
    |tls(TlsConfig::new()
        |root_certs(load_certs("ca.pem"))
        |identity(load_identity("client.pem", "client-key.pem"))
    )

// gRPC with mTLS
let channel = grpc::Channel::connect("https://server:50051")
    |tls(TlsConfig::new()
        |root_certs(load_certs("ca.pem"))
        |identity(load_identity("client.pem", "client-key.pem"))
        |client_auth(ClientAuth::Required)
    )
    |await
```

### 13.2 Authentication

```sigil
// Bearer token
let client = http::Client::new()
    |auth_bearer(token)

// Basic auth
let client = http::Client::new()
    |auth_basic(username, password)

// OAuth2
let client = http::Client::new()
    |oauth2(OAuth2Config {
        client_id: "...",
        client_secret: "...",
        token_url: "https://auth.example.com/token",
        scopes: ["read", "write"],
    })

// API key
let client = http::Client::new()
    |api_key("X-API-Key", key)
```

---

## 14. Performance Optimizations

### 14.1 Connection Pooling

```sigil
// HTTP connection pool
let client = http::Client::new()
    |pool_idle_timeout(90·sec)
    |pool_max_idle_per_host(10)
    |http2_keep_alive_interval(20·sec)

// gRPC channel with load balancing
let channel = grpc::Channel::connect("dns:///my-service:50051")
    |load_balance(LoadBalancer::RoundRobin)
    |keep_alive_timeout(20·sec)
```

### 14.2 Compression

```sigil
// HTTP compression
let client = http::Client::new()
    |accept_encoding([Encoding::Gzip, Encoding::Brotli])
    |request_compression(Encoding::Gzip)

// gRPC compression
let channel = grpc::Channel::connect(addr)
    |compression(Compression::Gzip)
```

### 14.3 Streaming Optimization

```sigil
// Buffered streaming
let stream = response|bytes_stream
    |buffer(64 * 1024)  // 64KB buffer

// Parallel processing of stream
let results = stream
    |par·τ(4){chunk => process(chunk)}
    |collect·await
```
