# AMQP 0-9-1 Protocol Implementation Specification

> *"Ethereal Whispers - messages that flow between realms"*

## 1. Overview

This specification defines the pure Sigil implementation of the AMQP 0-9-1 wire protocol,
as implemented by RabbitMQ. The implementation uses only Sigil's TCP socket primitives
(`TcpStream`) with no external dependencies.

### 1.1 Scope

| Feature | Priority | Status |
|---------|----------|--------|
| Connection handshake | P0 | **Implemented** |
| Channel management | P0 | **Implemented** |
| Basic publish | P0 | **Implemented** |
| Basic consume (push) | P1 | **Implemented** |
| Queue declare | P0 | **Implemented** |
| Queue bind | P0 | **Implemented** |
| Exchange declare | P0 | **Implemented** |
| Connection close | P1 | **Implemented** |
| Acknowledgments | P1 | **Implemented** |
| Transactions | P2 | Future |
| Publisher confirms | P2 | Future |
| TLS/SASL | P2 | Future |

### 1.3 Known Issues

**Method Call Syntax**: Static calls like `Channel·consume(ch, queue, tag)` may fail with
type errors. Use method call syntax instead: `ch.consume(queue, tag)`. This is a known
interpreter quirk where the static call dispatcher doesn't correctly pass the first argument
in all cases. The method call syntax works reliably.

**Queue Struct Passing**: When passing a Queue struct to `consume()`, use the queue name
string directly instead: `ch.consume("queue-name", "consumer-tag")`. Passing the Queue
struct returned from `declare_queue()` may cause type errors.

### 1.4 Implementation Status (2026-01-28)

**Implemented:**
- Full connection handshake (Start/Tune/Open)
- PLAIN SASL authentication with configurable credentials
- Address format: `user:pass@host:port` or `host:port` (defaults to guest/guest)
- Channel.Open/Open-Ok with auto-incrementing channel IDs
- Queue.Declare/Declare-Ok
- Queue.Bind/Bind-Ok for exchange routing
- Exchange.Declare/Declare-Ok (direct, fanout, topic, headers)
- Basic.Publish with content header and body frames
- Connection.Close/Close-Ok for graceful shutdown
- Channel.consume() - Start consuming via Basic.Consume (class=60, method=20)
- AmqpConsumer.next() - Wait for Basic.Deliver frames with timeout
- Delivery.ack() - Acknowledge delivery (Basic.Ack, class=60, method=80)
- Delivery.reject() - Reject delivery with requeue option (Basic.Reject, class=60, method=90)
- Delivery.nack() - Negative acknowledge (Basic.Nack, class=60, method=120 - RabbitMQ extension)
- AmqpConsumer.cancel() - Cancel consuming (Basic.Cancel, class=60, method=30)

**Not Yet Implemented:**
- Heartbeat frames (negotiated but not sent)
- Frame chunking for large messages
- Message properties (only body-size set)
- QoS/prefetch configuration

### 1.2 References

- [AMQP 0-9-1 Specification](https://www.rabbitmq.com/resources/specs/amqp0-9-1.pdf)
- [RabbitMQ Protocol Reference](https://www.rabbitmq.com/amqp-0-9-1-reference.html)
- Sigil Protocol Spec: `15-PROTOCOLS.md` § 8

---

## 2. Wire Protocol Format

### 2.1 Frame Structure

All AMQP frames follow this structure:

```
┌──────────┬─────────────┬──────────┬─────────────────┬────────────┐
│  Type    │  Channel    │   Size   │     Payload     │ Frame-End  │
│ (1 byte) │  (2 bytes)  │ (4 bytes)│  (Size bytes)   │  (1 byte)  │
│          │  big-endian │ big-end  │                 │   0xCE     │
└──────────┴─────────────┴──────────┴─────────────────┴────────────┘
```

### 2.2 Frame Types

| Type | Value | Description |
|------|-------|-------------|
| METHOD | 1 | AMQP method call |
| HEADER | 2 | Content header |
| BODY | 3 | Content body |
| HEARTBEAT | 8 | Heartbeat frame |

### 2.3 Method Frame Payload

```
MethodFrame => class_id method_id arguments
  class_id    => UINT16     // Class identifier
  method_id   => UINT16     // Method identifier
  arguments   => ...        // Method-specific arguments
```

### 2.4 Primitive Types

| Type | Encoding |
|------|----------|
| UINT8 | 1 byte unsigned |
| UINT16 | 2 bytes big-endian unsigned |
| UINT32 | 4 bytes big-endian unsigned |
| UINT64 | 8 bytes big-endian unsigned |
| SHORT_STRING | UINT8 length + UTF-8 bytes (max 255) |
| LONG_STRING | UINT32 length + bytes |
| FIELD_TABLE | UINT32 size + key-value pairs |
| TIMESTAMP | UINT64 (seconds since epoch) |

### 2.5 Field Table Encoding

Field tables are dictionaries with typed values:

```
FieldTable => size entries
  size     => UINT32        // Total size in bytes
  entries  => (key value)*
  key      => SHORT_STRING
  value    => type data
    type 't' => BOOLEAN (UINT8)
    type 'b' => INT8
    type 's' => INT16
    type 'I' => INT32
    type 'l' => INT64
    type 'f' => FLOAT32
    type 'd' => FLOAT64
    type 'S' => LONG_STRING
    type 'T' => TIMESTAMP
    type 'F' => FIELD_TABLE (nested)
    type 'A' => FIELD_ARRAY
    type 'V' => VOID (no data)
```

### 2.6 Key Classes and Methods

| Class | ID | Key Methods |
|-------|-----|-------------|
| Connection | 10 | start (10), start-ok (11), tune (30), tune-ok (31), open (40), open-ok (41), close (50) |
| Channel | 20 | open (10), open-ok (11), close (40), close-ok (41) |
| Exchange | 40 | declare (10), declare-ok (11), delete (20) |
| Queue | 50 | declare (10), declare-ok (11), bind (20), bind-ok (21), delete (40) |
| Basic | 60 | publish (40), consume (20), consume-ok (21), deliver (60), ack (80), reject (90), get (70), get-ok (71), get-empty (72) |

---

## 3. Sigil API Design

### 3.1 Module Structure

```sigil
protocol::amqp
├── Connection        // AMQP connection
├── Channel           // Communication channel
├── Queue             // Queue reference
├── Exchange          // Exchange reference
├── Message           // Message with properties
├── DeliveryMode      // Persistent/Transient
├── ExchangeType      // Direct/Fanout/Topic/Headers
├── ConnectionConfig  // Connection configuration
└── internal
    ├── Frame         // Frame encoding/decoding
    ├── Protocol      // Method handling
    └── FieldTable    // Field table codec
```

### 3.2 Connection API

```sigil
/// AMQP connection to broker
Σ Connection {
    stream: TcpStream,
    config: ConnectionConfig,
    channels: Map<u16, Channel>,
    next_channel: Cell<u16>,
    frame_max: u32,
    heartbeat: u16,
}

Σ ConnectionConfig {
    host: String,
    port: u16,
    username: String,
    password: String,
    vhost: String,
    heartbeat: u16,
    frame_max: u32,
}

impl Connection {
    /// Connect to AMQP broker
    async λ connect(config: ConnectionConfig) -> Result<Connection, AmqpError> {
        // 1. Establish TCP connection
        // 2. Send protocol header "AMQP\x00\x00\x09\x01"
        // 3. Receive Connection.Start, send Connection.Start-Ok
        // 4. Receive Connection.Tune, send Connection.Tune-Ok
        // 5. Send Connection.Open, receive Connection.Open-Ok
        // 6. Return connected connection
    }

    /// Create a new channel
    async λ create_channel(self) -> Result<Channel, AmqpError> {
        ≔ channel_id = self.next_channel.get();
        self.next_channel.set(channel_id + 1);

        // Send Channel.Open
        // Receive Channel.Open-Ok
        // Return channel
    }

    /// Close connection gracefully
    async λ close(self) -> Result<(), AmqpError> {
        // Send Connection.Close
        // Receive Connection.Close-Ok
        // Close TCP stream
    }
}
```

### 3.3 Channel API

```sigil
/// AMQP channel for messaging operations
Σ Channel {
    connection: &Connection,
    id: u16,
    consumers: Map<String, Consumer>,
}

impl Channel {
    /// Declare a queue
    async λ declare_queue(self, name: String, options: QueueOptions) -> Result<Queue, AmqpError> {
        // Send Queue.Declare
        // Receive Queue.Declare-Ok
        // Return queue reference
    }

    /// Declare an exchange
    async λ declare_exchange(self, name: String, exchange_type: ExchangeType, options: ExchangeOptions) -> Result<Exchange, AmqpError> {
        // Send Exchange.Declare
        // Receive Exchange.Declare-Ok
        // Return exchange reference
    }

    /// Bind queue to exchange
    async λ bind_queue(self, queue: &Queue, exchange: &Exchange, routing_key: String) -> Result<(), AmqpError> {
        // Send Queue.Bind
        // Receive Queue.Bind-Ok
    }

    /// Publish message to exchange
    async λ publish(self, exchange: String, routing_key: String, message: Message) -> Result<(), AmqpError> {
        // Send Basic.Publish method frame
        // Send content header frame
        // Send content body frame(s)
    }

    /// Start consuming from queue
    async λ consume(self, queue: &Queue, consumer_tag: String, options: ConsumeOptions) -> Result<Consumer, AmqpError> {
        // Send Basic.Consume
        // Receive Basic.Consume-Ok
        // Return consumer
    }

    /// Acknowledge message
    async λ ack(self, delivery_tag: u64, multiple: bool) -> Result<(), AmqpError> {
        // Send Basic.Ack
    }

    /// Reject message
    async λ reject(self, delivery_tag: u64, requeue: bool) -> Result<(), AmqpError> {
        // Send Basic.Reject
    }

    /// Close channel
    async λ close(self) -> Result<(), AmqpError> {
        // Send Channel.Close
        // Receive Channel.Close-Ok
    }
}
```

### 3.4 Queue and Exchange API

```sigil
/// Queue reference
Σ Queue {
    name: String,
    message_count: u32,
    consumer_count: u32,
}

Σ QueueOptions {
    durable: bool,       // Survive broker restart
    exclusive: bool,     // Only this connection can use
    auto_delete: bool,   // Delete when last consumer disconnects
    arguments: Map<String, Value>,
}

impl QueueOptions {
    λ default() -> QueueOptions {
        QueueOptions {
            durable: false,
            exclusive: false,
            auto_delete: false,
            arguments: Map·new(),
        }
    }

    λ durable() -> QueueOptions {
        QueueOptions { durable: true, ..QueueOptions·default() }
    }
}

/// Exchange reference
Σ Exchange {
    name: String,
    exchange_type: ExchangeType,
}

ε ExchangeType {
    Direct,    // Route by exact routing key match
    Fanout,    // Broadcast to all bound queues
    Topic,     // Route by pattern matching
    Headers,   // Route by header matching
}

Σ ExchangeOptions {
    durable: bool,
    auto_delete: bool,
    internal: bool,
    arguments: Map<String, Value>,
}
```

### 3.5 Message API

```sigil
/// AMQP message with properties
Σ Message {
    body: [u8],
    properties: MessageProperties,
}

Σ MessageProperties {
    content_type: Option<String>,
    content_encoding: Option<String>,
    headers: Option<Map<String, Value>>,
    delivery_mode: Option<DeliveryMode>,
    priority: Option<u8>,
    correlation_id: Option<String>,
    reply_to: Option<String>,
    expiration: Option<String>,
    message_id: Option<String>,
    timestamp: Option<u64>,
    type_: Option<String>,
    user_id: Option<String>,
    app_id: Option<String>,
}

ε DeliveryMode {
    Transient,   // 1: May be lost
    Persistent,  // 2: Written to disk
}

impl Message {
    /// Create message with body
    λ new(body: [u8]) -> Message {
        Message {
            body,
            properties: MessageProperties·default(),
        }
    }

    /// Create persistent message
    λ persistent(body: [u8]) -> Message {
        Message {
            body,
            properties: MessageProperties {
                delivery_mode: Some(DeliveryMode·Persistent),
                ..MessageProperties·default()
            },
        }
    }

    /// Set content type
    λ with_content_type(self, content_type: String) -> Message {
        Message {
            properties: MessageProperties {
                content_type: Some(content_type),
                ..self.properties
            },
            ..self
        }
    }

    /// Set correlation ID
    λ with_correlation_id(self, correlation_id: String) -> Message {
        Message {
            properties: MessageProperties {
                correlation_id: Some(correlation_id),
                ..self.properties
            },
            ..self
        }
    }

    /// Set reply-to
    λ with_reply_to(self, reply_to: String) -> Message {
        Message {
            properties: MessageProperties {
                reply_to: Some(reply_to),
                ..self.properties
            },
            ..self
        }
    }
}
```

### 3.6 Consumer API

```sigil
/// Message consumer
Σ Consumer {
    tag: String,
    channel: &Channel,
    queue: String,
}

/// Delivered message
Σ Delivery {
    consumer_tag: String,
    delivery_tag: u64,
    redelivered: bool,
    exchange: String,
    routing_key: String,
    message: Message,
}

Σ ConsumeOptions {
    no_local: bool,     // Don't receive own messages
    no_ack: bool,       // Auto-acknowledge
    exclusive: bool,    // Exclusive consumer
    no_wait: bool,      // Don't wait for server response
    arguments: Map<String, Value>,
}

impl Consumer {
    /// Receive next delivery (blocking)
    async λ next(self) -> Result<Option<Delivery>, AmqpError> {
        // Wait for Basic.Deliver frame
        // Read content header and body frames
        // Return delivery
    }

    /// Cancel consumer
    async λ cancel(self) -> Result<(), AmqpError> {
        // Send Basic.Cancel
        // Receive Basic.Cancel-Ok
    }
}
```

### 3.7 Error Types

```sigil
ε AmqpError {
    ConnectionFailed(String),
    ConnectionClosed { code: u16, reason: String },
    ChannelClosed { code: u16, reason: String },
    FrameError(String),
    Timeout,
    InvalidFrame,
    UnexpectedFrame { expected: String, got: String },
    AuthenticationFailed,
    VhostAccessRefused,
    ResourceLocked,
    NotFound,
    AccessRefused,
    PreconditionFailed(String),
}

// AMQP reply codes mapping
λ error_from_reply(code: u16, text: String) -> AmqpError {
    ⌥ code {
        200 => panic!("Reply success"),
        311 => AmqpError·ContentTooLarge,
        312 => AmqpError·NoRoute,
        313 => AmqpError·NoConsumers,
        320 => AmqpError·ConnectionForced { code, reason: text },
        402 => AmqpError·InvalidPath,
        403 => AmqpError·AccessRefused,
        404 => AmqpError·NotFound,
        405 => AmqpError·ResourceLocked,
        406 => AmqpError·PreconditionFailed(text),
        501 => AmqpError·FrameError(text),
        502 => AmqpError·SyntaxError,
        503 => AmqpError·CommandInvalid,
        504 => AmqpError·ChannelError,
        505 => AmqpError·UnexpectedFrame { expected: "", got: text },
        530 => AmqpError·NotAllowed,
        540 => AmqpError·NotImplemented,
        541 => AmqpError·InternalError,
        _ => AmqpError·Unknown { code, reason: text },
    }
}
```

---

## 4. Internal Implementation

### 4.1 Frame Codec

```sigil
/// Internal frame encoding/decoding
mod internal·Frame {
    // Frame type constants
    const FRAME_METHOD: u8 = 1;
    const FRAME_HEADER: u8 = 2;
    const FRAME_BODY: u8 = 3;
    const FRAME_HEARTBEAT: u8 = 8;
    const FRAME_END: u8 = 0xCE;

    /// Encode a method frame
    λ encode_method(channel: u16, class_id: u16, method_id: u16, arguments: [u8]) -> [u8] {
        ≔ payload = [
            ..encode_uint16(class_id),
            ..encode_uint16(method_id),
            ..arguments,
        ];

        ≔ size = payload.len() as u32;

        [
            FRAME_METHOD,
            ..encode_uint16(channel),
            ..encode_uint32(size),
            ..payload,
            FRAME_END,
        ]
    }

    /// Encode a content header frame
    λ encode_header(channel: u16, class_id: u16, body_size: u64, properties: MessageProperties) -> [u8] {
        ≔ property_flags = compute_property_flags(properties);
        ≔ property_list = encode_properties(properties);

        ≔ payload = [
            ..encode_uint16(class_id),
            ..encode_uint16(0),  // weight (reserved)
            ..encode_uint64(body_size),
            ..encode_uint16(property_flags),
            ..property_list,
        ];

        ≔ size = payload.len() as u32;

        [
            FRAME_HEADER,
            ..encode_uint16(channel),
            ..encode_uint32(size),
            ..payload,
            FRAME_END,
        ]
    }

    /// Encode a body frame
    λ encode_body(channel: u16, body: [u8]) -> [u8] {
        ≔ size = body.len() as u32;

        [
            FRAME_BODY,
            ..encode_uint16(channel),
            ..encode_uint32(size),
            ..body,
            FRAME_END,
        ]
    }

    /// Encode heartbeat frame
    λ encode_heartbeat() -> [u8] {
        [
            FRAME_HEARTBEAT,
            0, 0,  // channel 0
            0, 0, 0, 0,  // size 0
            FRAME_END,
        ]
    }

    /// Decode a frame from bytes
    λ decode(data: [u8], offset: &mut usize) -> Result<Frame, AmqpError> {
        ≔ frame_type = data[*offset];
        ≔ channel = decode_uint16(data, &mut (*offset + 1));
        ≔ size = decode_uint32(data, &mut (*offset + 3));
        ≔ payload = data[*offset + 7..*offset + 7 + size as usize];
        ≔ frame_end = data[*offset + 7 + size as usize];

        if frame_end != FRAME_END {
            return Err(AmqpError·FrameError("Invalid frame end marker"));
        }

        *offset += 8 + size as usize;

        ⌥ frame_type {
            FRAME_METHOD => decode_method_frame(channel, payload),
            FRAME_HEADER => decode_header_frame(channel, payload),
            FRAME_BODY => Ok(Frame·Body { channel, data: payload }),
            FRAME_HEARTBEAT => Ok(Frame·Heartbeat),
            _ => Err(AmqpError·InvalidFrame),
        }
    }
}
```

### 4.2 Primitive Encoding

```sigil
mod internal·Codec {
    // === Encoding ===

    λ encode_uint8(value: u8) -> [u8; 1] {
        [value]
    }

    λ encode_uint16(value: u16) -> [u8; 2] {
        value|to_be_bytes
    }

    λ encode_uint32(value: u32) -> [u8; 4] {
        value|to_be_bytes
    }

    λ encode_uint64(value: u64) -> [u8; 8] {
        value|to_be_bytes
    }

    λ encode_short_string(s: String) -> [u8] {
        ≔ bytes = s|as_bytes;
        ≔ len = min(bytes.len(), 255);
        [len as u8, ..bytes[0..len]]
    }

    λ encode_long_string(s: String) -> [u8] {
        ≔ bytes = s|as_bytes;
        [..encode_uint32(bytes.len() as u32), ..bytes]
    }

    λ encode_field_table(table: Map<String, Value>) -> [u8] {
        ≔ entries = table|iter|flat_map(λ(key, value) {
            ≔ key_bytes = encode_short_string(key);
            ≔ value_bytes = encode_field_value(value);
            [..key_bytes, ..value_bytes]
        })|collect;

        [..encode_uint32(entries.len() as u32), ..entries]
    }

    λ encode_field_value(value: Value) -> [u8] {
        ⌥ value {
            Value·Bool(b) => ['t', if b { 1 } else { 0 }],
            Value·Int8(n) => ['b', n as u8],
            Value·Int16(n) => ['s', ..encode_uint16(n as u16)],
            Value·Int32(n) => ['I', ..encode_uint32(n as u32)],
            Value·Int64(n) => ['l', ..encode_uint64(n as u64)],
            Value·Float(f) => ['f', ..f|to_be_bytes],
            Value·Double(d) => ['d', ..d|to_be_bytes],
            Value·String(s) => ['S', ..encode_long_string(s)],
            Value·Timestamp(t) => ['T', ..encode_uint64(t)],
            Value·Table(t) => ['F', ..encode_field_table(t)],
            Value·Void => ['V'],
        }
    }

    // === Decoding ===

    λ decode_uint16(data: [u8], offset: &mut usize) -> u16 {
        ≔ bytes = [data[*offset], data[*offset + 1]];
        *offset += 2;
        u16·from_be_bytes(bytes)
    }

    λ decode_uint32(data: [u8], offset: &mut usize) -> u32 {
        ≔ bytes = [data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]];
        *offset += 4;
        u32·from_be_bytes(bytes)
    }

    λ decode_uint64(data: [u8], offset: &mut usize) -> u64 {
        ≔ result = u64·from_be_bytes(data[*offset..*offset + 8]);
        *offset += 8;
        result
    }

    λ decode_short_string(data: [u8], offset: &mut usize) -> String {
        ≔ len = data[*offset] as usize;
        *offset += 1;
        ≔ bytes = data[*offset..*offset + len];
        *offset += len;
        String·from_utf8(bytes)|unwrap_or("")
    }

    λ decode_long_string(data: [u8], offset: &mut usize) -> String {
        ≔ len = decode_uint32(data, offset) as usize;
        ≔ bytes = data[*offset..*offset + len];
        *offset += len;
        String·from_utf8(bytes)|unwrap_or("")
    }
}
```

### 4.3 Connection Handshake

```sigil
mod internal·Handshake {
    // Protocol header: "AMQP" + 0 + 0 + 9 + 1
    const PROTOCOL_HEADER: [u8; 8] = ['A', 'M', 'Q', 'P', 0, 0, 9, 1];

    /// Perform AMQP connection handshake
    async λ handshake(stream: &TcpStream, config: &ConnectionConfig) -> Result<(u32, u16), AmqpError> {
        // Step 1: Send protocol header
        stream|write_all(PROTOCOL_HEADER)|await?;
        stream|flush|await?;

        // Step 2: Receive Connection.Start
        ≔ start = read_method(stream)|await?;
        if start.class_id != 10 || start.method_id != 10 {
            return Err(AmqpError·UnexpectedFrame {
                expected: "Connection.Start",
                got: "{start.class_id}.{start.method_id}",
            });
        }

        // Parse Connection.Start
        ≔ version_major = start.arguments[0];
        ≔ version_minor = start.arguments[1];
        // ... parse server properties, mechanisms, locales

        // Step 3: Send Connection.Start-Ok
        ≔ start_ok = encode_connection_start_ok(config);
        stream|write_all(start_ok)|await?;

        // Step 4: Receive Connection.Tune
        ≔ tune = read_method(stream)|await?;
        if tune.class_id != 10 || tune.method_id != 30 {
            return Err(AmqpError·UnexpectedFrame {
                expected: "Connection.Tune",
                got: "{tune.class_id}.{tune.method_id}",
            });
        }

        // Parse tune parameters
        ≔ channel_max = decode_uint16(tune.arguments, &mut 0);
        ≔ frame_max = decode_uint32(tune.arguments, &mut 2);
        ≔ heartbeat = decode_uint16(tune.arguments, &mut 6);

        // Negotiate values
        ≔ negotiated_frame_max = if config.frame_max == 0 { frame_max } else { min(config.frame_max, frame_max) };
        ≔ negotiated_heartbeat = if config.heartbeat == 0 { heartbeat } else { min(config.heartbeat, heartbeat) };

        // Step 5: Send Connection.Tune-Ok
        ≔ tune_ok = encode_connection_tune_ok(channel_max, negotiated_frame_max, negotiated_heartbeat);
        stream|write_all(tune_ok)|await?;

        // Step 6: Send Connection.Open
        ≔ open = encode_connection_open(config.vhost);
        stream|write_all(open)|await?;

        // Step 7: Receive Connection.Open-Ok
        ≔ open_ok = read_method(stream)|await?;
        if open_ok.class_id != 10 || open_ok.method_id != 41 {
            return Err(AmqpError·UnexpectedFrame {
                expected: "Connection.Open-Ok",
                got: "{open_ok.class_id}.{open_ok.method_id}",
            });
        }

        Ok((negotiated_frame_max, negotiated_heartbeat))
    }

    λ encode_connection_start_ok(config: &ConnectionConfig) -> [u8] {
        ≔ client_properties = encode_field_table(Map·from([
            ("product", Value·String("sigil-amqp")),
            ("version", Value·String("0.1.0")),
            ("platform", Value·String("Sigil")),
        ]));

        ≔ mechanism = encode_short_string("PLAIN");
        ≔ response = encode_long_string("\x00{config.username}\x00{config.password}");
        ≔ locale = encode_short_string("en_US");

        ≔ arguments = [..client_properties, ..mechanism, ..response, ..locale];

        Frame·encode_method(0, 10, 11, arguments)  // Connection.Start-Ok
    }

    λ encode_connection_tune_ok(channel_max: u16, frame_max: u32, heartbeat: u16) -> [u8] {
        ≔ arguments = [
            ..encode_uint16(channel_max),
            ..encode_uint32(frame_max),
            ..encode_uint16(heartbeat),
        ];

        Frame·encode_method(0, 10, 31, arguments)  // Connection.Tune-Ok
    }

    λ encode_connection_open(vhost: String) -> [u8] {
        ≔ arguments = [
            ..encode_short_string(vhost),
            0,  // reserved
            0,  // reserved
        ];

        Frame·encode_method(0, 10, 40, arguments)  // Connection.Open
    }
}
```

### 4.4 Method Encoding

```sigil
mod internal·Methods {
    // === Channel ===

    λ encode_channel_open(channel: u16) -> [u8] {
        ≔ arguments = encode_short_string("");  // reserved
        Frame·encode_method(channel, 20, 10, arguments)
    }

    λ encode_channel_close(channel: u16, code: u16, reason: String, class_id: u16, method_id: u16) -> [u8] {
        ≔ arguments = [
            ..encode_uint16(code),
            ..encode_short_string(reason),
            ..encode_uint16(class_id),
            ..encode_uint16(method_id),
        ];
        Frame·encode_method(channel, 20, 40, arguments)
    }

    // === Queue ===

    λ encode_queue_declare(channel: u16, name: String, options: QueueOptions) -> [u8] {
        ≔ flags = 0u8;
        if options.passive { flags |= 0x01; }
        if options.durable { flags |= 0x02; }
        if options.exclusive { flags |= 0x04; }
        if options.auto_delete { flags |= 0x08; }
        if options.no_wait { flags |= 0x10; }

        ≔ arguments = [
            ..encode_uint16(0),  // reserved
            ..encode_short_string(name),
            flags,
            ..encode_field_table(options.arguments),
        ];

        Frame·encode_method(channel, 50, 10, arguments)
    }

    λ encode_queue_bind(channel: u16, queue: String, exchange: String, routing_key: String, arguments: Map<String, Value>) -> [u8] {
        ≔ args = [
            ..encode_uint16(0),  // reserved
            ..encode_short_string(queue),
            ..encode_short_string(exchange),
            ..encode_short_string(routing_key),
            0,  // no-wait = false
            ..encode_field_table(arguments),
        ];

        Frame·encode_method(channel, 50, 20, args)
    }

    // === Exchange ===

    λ encode_exchange_declare(channel: u16, name: String, exchange_type: ExchangeType, options: ExchangeOptions) -> [u8] {
        ≔ type_name = ⌥ exchange_type {
            ExchangeType·Direct => "direct",
            ExchangeType·Fanout => "fanout",
            ExchangeType·Topic => "topic",
            ExchangeType·Headers => "headers",
        };

        ≔ flags = 0u8;
        if options.passive { flags |= 0x01; }
        if options.durable { flags |= 0x02; }
        if options.auto_delete { flags |= 0x04; }
        if options.internal { flags |= 0x08; }
        if options.no_wait { flags |= 0x10; }

        ≔ arguments = [
            ..encode_uint16(0),  // reserved
            ..encode_short_string(name),
            ..encode_short_string(type_name),
            flags,
            ..encode_field_table(options.arguments),
        ];

        Frame·encode_method(channel, 40, 10, arguments)
    }

    // === Basic ===

    λ encode_basic_publish(channel: u16, exchange: String, routing_key: String, mandatory: bool, immediate: bool) -> [u8] {
        ≔ flags = 0u8;
        if mandatory { flags |= 0x01; }
        if immediate { flags |= 0x02; }

        ≔ arguments = [
            ..encode_uint16(0),  // reserved
            ..encode_short_string(exchange),
            ..encode_short_string(routing_key),
            flags,
        ];

        Frame·encode_method(channel, 60, 40, arguments)
    }

    λ encode_basic_consume(channel: u16, queue: String, consumer_tag: String, options: ConsumeOptions) -> [u8] {
        ≔ flags = 0u8;
        if options.no_local { flags |= 0x01; }
        if options.no_ack { flags |= 0x02; }
        if options.exclusive { flags |= 0x04; }
        if options.no_wait { flags |= 0x08; }

        ≔ arguments = [
            ..encode_uint16(0),  // reserved
            ..encode_short_string(queue),
            ..encode_short_string(consumer_tag),
            flags,
            ..encode_field_table(options.arguments),
        ];

        Frame·encode_method(channel, 60, 20, arguments)
    }

    λ encode_basic_ack(channel: u16, delivery_tag: u64, multiple: bool) -> [u8] {
        ≔ arguments = [
            ..encode_uint64(delivery_tag),
            if multiple { 1 } else { 0 },
        ];

        Frame·encode_method(channel, 60, 80, arguments)
    }

    λ encode_basic_reject(channel: u16, delivery_tag: u64, requeue: bool) -> [u8] {
        ≔ arguments = [
            ..encode_uint64(delivery_tag),
            if requeue { 1 } else { 0 },
        ];

        Frame·encode_method(channel, 60, 90, arguments)
    }
}
```

---

## 5. Implementation Phases

### Phase 1: Foundation (P0)
1. Primitive encoding/decoding (uint8/16/32/64, strings, field tables)
2. Frame encoding/decoding (method, header, body)
3. Connection handshake (Start/Tune/Open)
4. Basic error handling

### Phase 2: Core Messaging (P0)
1. Channel open/close
2. Queue declare
3. Basic publish (method + header + body frames)
4. Integration with Docker test infrastructure

### Phase 3: Exchanges and Bindings (P1)
1. Exchange declare (direct, fanout, topic)
2. Queue bind
3. Routing key support

### Phase 4: Consuming (P1)
1. Basic consume (push-based delivery)
2. Basic.Deliver handling
3. Acknowledgments (ack, reject, nack)
4. Consumer cancellation

### Phase 5: Hardening (P2)
1. Heartbeat support
2. Frame chunking for large messages
3. Connection recovery
4. Proper property encoding (all 14 basic properties)

### Phase 6: Edge Cases (P2 - Future)

The following edge cases are deferred for future hardening work:

| Edge Case | Priority | Description |
|-----------|----------|-------------|
| Multiple deliveries | P2 | Consume multiple messages in sequence |
| Requeue behavior | P2 | Verify reject with requeue=true returns message to queue |
| Consumer cancellation mid-stream | P2 | Graceful shutdown while messages in flight |
| Large messages | P2 | Messages exceeding frame_max requiring chunking |
| Connection timeout | P2 | Graceful handling of broker connection failures |
| Broker failover | P2 | Reconnection logic when broker becomes unavailable |
| Channel errors | P2 | Handle channel-level exceptions without closing connection |
| QoS/prefetch | P2 | Implement Basic.Qos for flow control |

---

## 6. Test Cases

### 6.1 Unit Tests

```sigil
// Test primitive encoding
λ test_encode_short_string() {
    assert_eq(Codec·encode_short_string("hello"), [5, 'h', 'e', 'l', 'l', 'o']);
    assert_eq(Codec·encode_short_string(""), [0]);
}

λ test_encode_field_table() {
    ≔ table = Map·from([
        ("key", Value·String("value")),
    ]);
    ≔ encoded = Codec·encode_field_table(table);
    // size (4) + key "key" (1+3) + 'S' (1) + string length (4) + "value" (5) = 18
    assert_eq(encoded.len(), 18);
}

λ test_frame_encoding() {
    ≔ frame = Frame·encode_method(1, 50, 10, []);  // Queue.Declare on channel 1
    assert_eq(frame[0], 1);  // METHOD frame type
    assert_eq(frame[7], 0);  // Frame end marker should be at correct position
}
```

### 6.2 Integration Tests

```sigil
// Requires: ./infrastructure/infra.sh up

async λ test_connection() {
    ≔ config = ConnectionConfig {
        host: "localhost",
        port: 5672,
        username: "sigil",
        password: "sigil",
        vhost: "/",
        heartbeat: 60,
        frame_max: 131072,
    };

    ≔ conn = Connection·connect(config)|await!;
    assert(conn.frame_max > 0);

    conn.close()|await!;
}

async λ test_queue_declare() {
    ≔ conn = connect_default()|await!;
    ≔ channel = conn.create_channel()|await!;

    ≔ queue = channel.declare_queue("test-queue", QueueOptions·default())|await!;
    assert_eq(queue.name, "test-queue");

    channel.close()|await!;
    conn.close()|await!;
}

async λ test_publish_consume() {
    ≔ conn = connect_default()|await!;
    ≔ channel = conn.create_channel()|await!;

    // Declare queue
    ≔ queue = channel.declare_queue("test-pubsub", QueueOptions·default())|await!;

    // Publish message
    ≔ msg = Message·new("hello amqp"|as_bytes);
    channel.publish("", "test-pubsub", msg)|await!;

    // Consume message
    ≔ consumer = channel.consume(&queue, "test-consumer", ConsumeOptions·default())|await!;
    ≔ delivery = consumer.next()|await!;

    assert_eq(delivery.unwrap().message.body, "hello amqp"|as_bytes);

    consumer.cancel()|await!;
    channel.close()|await!;
    conn.close()|await!;
}

async λ test_fanout_exchange() {
    ≔ conn = connect_default()|await!;
    ≔ channel = conn.create_channel()|await!;

    // Declare fanout exchange
    ≔ exchange = channel.declare_exchange("test-fanout", ExchangeType·Fanout, ExchangeOptions·default())|await!;

    // Declare and bind two queues
    ≔ q1 = channel.declare_queue("fanout-q1", QueueOptions·default())|await!;
    ≔ q2 = channel.declare_queue("fanout-q2", QueueOptions·default())|await!;
    channel.bind_queue(&q1, &exchange, "")|await!;
    channel.bind_queue(&q2, &exchange, "")|await!;

    // Publish to exchange
    ≔ msg = Message·new("broadcast"|as_bytes);
    channel.publish("test-fanout", "", msg)|await!;

    // Both queues should receive the message
    // ... consume and verify

    channel.close()|await!;
    conn.close()|await!;
}
```

### 6.3 Acceptance Criteria

| Test | Criteria |
|------|----------|
| Connection | Successfully connect with PLAIN auth |
| Channel | Open and close channels |
| Queue | Declare queue, verify message/consumer counts |
| Publish | Send message, no errors |
| Consume | Receive published message |
| Ack | Acknowledge removes message from queue |
| Exchange | Fanout broadcasts to all bound queues |
| Topic | Pattern matching routes correctly |
| Heartbeat | Connection stays alive with heartbeats |

---

## 7. Security Considerations

### 7.1 Current Scope (PLAIN Auth)

Initial implementation supports:
- PLAIN SASL mechanism (username/password in cleartext)
- Suitable for local development and trusted networks

### 7.2 Future: SASL Mechanisms

```sigil
ε SaslMechanism {
    Plain,
    External,      // TLS client certificate
    AmqPlain,      // Legacy RabbitMQ
}
```

### 7.3 Future: TLS

```sigil
Σ TlsConfig {
    ca_cert: Option<String>,
    client_cert: Option<String>,
    client_key: Option<String>,
    verify_hostname: bool,
}
```

---

## 8. Performance Considerations

### 8.1 Frame Size

- Default frame_max: 131072 (128KB)
- Large messages split across multiple body frames
- Configurable per-connection

### 8.2 Prefetch

```sigil
// Future: QoS configuration
async λ set_qos(channel: &Channel, prefetch_count: u16) -> Result<(), AmqpError> {
    // Basic.Qos
}
```

### 8.3 Publisher Confirms (P2)

```sigil
// Future: reliable publishing
async λ confirm_select(channel: &Channel) -> Result<(), AmqpError> {
    // Confirm.Select
}

async λ wait_for_confirms(channel: &Channel) -> Result<(), AmqpError> {
    // Wait for all confirms
}
```

---

## Appendix A: AMQP 0-9-1 Method Reference

### A.1 Connection Class (10)

| Method | ID | Direction |
|--------|-----|-----------|
| Start | 10 | S→C |
| Start-Ok | 11 | C→S |
| Secure | 20 | S→C |
| Secure-Ok | 21 | C→S |
| Tune | 30 | S→C |
| Tune-Ok | 31 | C→S |
| Open | 40 | C→S |
| Open-Ok | 41 | S→C |
| Close | 50 | Both |
| Close-Ok | 51 | Both |

### A.2 Channel Class (20)

| Method | ID | Direction |
|--------|-----|-----------|
| Open | 10 | C→S |
| Open-Ok | 11 | S→C |
| Flow | 20 | Both |
| Flow-Ok | 21 | Both |
| Close | 40 | Both |
| Close-Ok | 41 | Both |

### A.3 Queue Class (50)

| Method | ID | Direction |
|--------|-----|-----------|
| Declare | 10 | C→S |
| Declare-Ok | 11 | S→C |
| Bind | 20 | C→S |
| Bind-Ok | 21 | S→C |
| Unbind | 50 | C→S |
| Unbind-Ok | 51 | S→C |
| Purge | 30 | C→S |
| Purge-Ok | 31 | S→C |
| Delete | 40 | C→S |
| Delete-Ok | 41 | S→C |

### A.4 Basic Class (60)

| Method | ID | Direction |
|--------|-----|-----------|
| Qos | 10 | C→S |
| Qos-Ok | 11 | S→C |
| Consume | 20 | C→S |
| Consume-Ok | 21 | S→C |
| Cancel | 30 | C→S |
| Cancel-Ok | 31 | S→C |
| Publish | 40 | C→S |
| Return | 50 | S→C |
| Deliver | 60 | S→C |
| Get | 70 | C→S |
| Get-Ok | 71 | S→C |
| Get-Empty | 72 | S→C |
| Ack | 80 | Both |
| Reject | 90 | C→S |
| Recover | 110 | C→S |
| Recover-Ok | 111 | S→C |
| Nack | 120 | Both |

---

## Appendix B: Content Properties

Basic class content properties (property flags):

| Bit | Property | Type |
|-----|----------|------|
| 15 | content-type | shortstr |
| 14 | content-encoding | shortstr |
| 13 | headers | table |
| 12 | delivery-mode | octet |
| 11 | priority | octet |
| 10 | correlation-id | shortstr |
| 9 | reply-to | shortstr |
| 8 | expiration | shortstr |
| 7 | message-id | shortstr |
| 6 | timestamp | timestamp |
| 5 | type | shortstr |
| 4 | user-id | shortstr |
| 3 | app-id | shortstr |
| 2 | reserved | shortstr |
