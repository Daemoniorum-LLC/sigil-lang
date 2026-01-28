# Kafka Protocol Implementation Specification

> *"Ancestral Memory - messages that persist across time"*

## 1. Overview

This specification defines the pure Sigil implementation of the Apache Kafka wire protocol.
The implementation uses only Sigil's TCP socket primitives (`TcpStream`) with no external
dependencies.

### 1.1 Scope

| Feature | Priority | Status |
|---------|----------|--------|
| Producer (single message) | P0 | **Implemented** |
| Producer (batch) | P1 | Specified |
| Consumer (poll) | P1 | **Implemented** |
| Metadata discovery | P1 | Specified |
| Consumer groups | P2 | Future |
| Transactions | P2 | Future |
| SASL authentication | P2 | Future |

### 1.3 Known Issues

**Method Call Syntax**: Static calls like `Consumer·subscribe(consumer, topics)` may fail with
type errors. Use method call syntax instead: `consumer.subscribe(topics)`. This is a known
interpreter quirk where the static call dispatcher doesn't correctly pass the first argument
in all cases. The method call syntax works reliably.

### 1.4 Implementation Status (2026-01-28)

**Implemented:**
- TCP connection with length-prefix framing
- ApiVersions request/response (API key 18, version 0)
- Produce request/response (API key 0, version 3)
- RecordBatch v2 encoding with CRC32-C checksum
- Correlation ID tracking (auto-incremented)
- Error code parsing in Produce response
- Producer.close() for cleanup
- Consumer.connect() - Connect to Kafka broker as consumer
- Consumer.subscribe() - Subscribe to topic list
- Consumer.poll() - Fetch messages using Kafka Fetch API (key 1, version 4)
- Consumer.close() - Close consumer connection
- RecordBatch v2 decoding with zig-zag varint parsing

**Not Yet Implemented:**
- Metadata request (cluster discovery)
- Batch sending (multiple records)
- Compression (gzip, snappy, lz4, zstd)
- Connection pooling
- Consumer groups (JoinGroup, SyncGroup, Heartbeat)

### 1.2 References

- [Kafka Protocol Guide](https://kafka.apache.org/protocol.html)
- [KIP-482: API Versions](https://cwiki.apache.org/confluence/display/KAFKA/KIP-482)
- Sigil Protocol Spec: `15-PROTOCOLS.md` § 7

---

## 2. Wire Protocol Format

### 2.1 Message Framing

All Kafka messages are length-prefixed with a 4-byte big-endian integer:

```
┌─────────────────┬────────────────────────────────┐
│  Size (4 bytes) │        Message Body            │
│   big-endian    │     (Size bytes)               │
└─────────────────┴────────────────────────────────┘
```

### 2.2 Request Header (v2)

```
RequestHeader => api_key api_version correlation_id client_id
  api_key         => INT16      // API identifier
  api_version     => INT16      // API version
  correlation_id  => INT32      // Client-generated request ID
  client_id       => NULLABLE_STRING
```

### 2.3 Response Header (v1)

```
ResponseHeader => correlation_id
  correlation_id  => INT32      // Matches request correlation_id
```

### 2.4 Primitive Types

| Type | Encoding |
|------|----------|
| INT8 | 1 byte signed |
| INT16 | 2 bytes big-endian signed |
| INT32 | 4 bytes big-endian signed |
| INT64 | 8 bytes big-endian signed |
| VARINT | Variable-length zig-zag encoded |
| STRING | INT16 length + UTF-8 bytes |
| NULLABLE_STRING | INT16 length (-1 = null) + UTF-8 bytes |
| BYTES | INT32 length + raw bytes |
| NULLABLE_BYTES | INT32 length (-1 = null) + raw bytes |
| ARRAY | INT32 count + elements |
| COMPACT_STRING | VARINT length + UTF-8 bytes (v2+) |
| COMPACT_ARRAY | VARINT count + elements (v2+) |

### 2.5 Key API Operations

| API Key | Name | Min Version | Description |
|---------|------|-------------|-------------|
| 0 | Produce | 0 | Send messages to topic |
| 1 | Fetch | 0 | Retrieve messages from topic |
| 3 | Metadata | 0 | Discover brokers and topics |
| 18 | ApiVersions | 0 | Query supported API versions |

---

## 3. Sigil API Design

### 3.1 Module Structure

```sigil
protocol::kafka
├── Producer          // Message producer
├── Consumer          // Message consumer
├── Record            // Message record
├── Topic             // Topic reference
├── ProducerConfig    // Producer configuration
├── ConsumerConfig    // Consumer configuration
├── Acks              // Acknowledgment mode
├── Compression       // Compression codec
└── internal
    ├── Connection    // TCP connection wrapper
    ├── Codec         // Binary encoding/decoding
    └── Protocol      // Request/response handling
```

### 3.2 Producer API

```sigil
/// Kafka producer for sending messages
Σ Producer {
    connection: internal·Connection,
    config: ProducerConfig,
    correlation_id: Cell<i32>,
}

Σ ProducerConfig {
    bootstrap_servers: [String],
    acks: Acks,
    compression: Compression,
    client_id: String,
    request_timeout_ms: i32,
}

ε Acks {
    None,      // acks=0: No acknowledgment
    Leader,    // acks=1: Leader acknowledgment only
    All,       // acks=-1: All in-sync replicas
}

ε Compression {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

impl Producer {
    /// Connect to Kafka cluster
    async λ connect(config: ProducerConfig) -> Result<Producer, KafkaError> {
        // 1. Resolve bootstrap server
        // 2. Establish TCP connection
        // 3. Send ApiVersions request
        // 4. Send Metadata request to discover cluster
        // 5. Return connected producer
    }

    /// Send a single record
    async λ send(self, record: Record) -> Result<RecordMetadata, KafkaError> {
        // 1. Encode record into RecordBatch
        // 2. Build Produce request
        // 3. Send request, await response
        // 4. Parse response, return metadata
    }

    /// Send batch of records
    async λ send_batch(self, records: [Record]) -> Result<[RecordMetadata], KafkaError> {
        // 1. Group records by topic-partition
        // 2. Encode each group into RecordBatch
        // 3. Build Produce request with all batches
        // 4. Send request, await response
        // 5. Parse response, return metadata
    }

    /// Close producer connection
    async λ close(self) -> Result<(), KafkaError> {
        self.connection.close()
    }
}
```

### 3.3 Record API

```sigil
/// A Kafka record (message)
Σ Record {
    topic: String,
    partition: Option<i32>,    // None = use partitioner
    key: Option<[u8]>,
    value: [u8],
    headers: [(String, [u8])],
    timestamp: Option<i64>,    // None = broker assigns
}

impl Record {
    /// Create record with topic and value
    λ new(topic: String, value: [u8]) -> Record {
        Record {
            topic,
            partition: None,
            key: None,
            value,
            headers: [],
            timestamp: None,
        }
    }

    /// Set record key
    λ with_key(self, key: [u8]) -> Record {
        Record { key: Some(key), ..self }
    }

    /// Set record headers
    λ with_headers(self, headers: [(String, [u8])]) -> Record {
        Record { headers, ..self }
    }
}

/// Metadata returned after successful send
Σ RecordMetadata {
    topic: String,
    partition: i32,
    offset: i64,
    timestamp: i64,
}
```

### 3.4 Consumer API

```sigil
/// Kafka consumer for receiving messages
Σ Consumer {
    connection: internal·Connection,
    config: ConsumerConfig,
    subscriptions: [String],
    offsets: Map<(String, i32), i64>,  // (topic, partition) -> offset
}

Σ ConsumerConfig {
    bootstrap_servers: [String],
    group_id: Option<String>,
    auto_offset_reset: OffsetReset,
    client_id: String,
    fetch_max_bytes: i32,
    fetch_max_wait_ms: i32,
}

ε OffsetReset {
    Earliest,
    Latest,
    None,
}

impl Consumer {
    /// Connect to Kafka cluster
    async λ connect(config: ConsumerConfig) -> Result<Consumer, KafkaError> {
        // 1. Resolve bootstrap server
        // 2. Establish TCP connection
        // 3. Send Metadata request
        // 4. Return connected consumer
    }

    /// Subscribe to topics
    λ subscribe(self, topics: [String]) -> Result<(), KafkaError> {
        self.subscriptions = topics;
        Ok(())
    }

    /// Poll for records
    async λ poll(self, timeout_ms: i32) -> Result<[ConsumerRecord], KafkaError> {
        // 1. Build Fetch request for subscribed topics
        // 2. Send request with timeout
        // 3. Parse response into records
        // 4. Update offsets
        // 5. Return records
    }

    /// Commit current offsets
    async λ commit(self) -> Result<(), KafkaError> {
        // Build and send OffsetCommit request
    }
}

/// Record received from consumer
Σ ConsumerRecord {
    topic: String,
    partition: i32,
    offset: i64,
    timestamp: i64,
    key: Option<[u8]>,
    value: [u8],
    headers: [(String, [u8])],
}
```

### 3.5 Error Types

```sigil
ε KafkaError {
    ConnectionFailed(String),
    Timeout,
    InvalidResponse,
    BrokerError { code: i16, message: String },
    UnknownTopicOrPartition,
    LeaderNotAvailable,
    NotLeaderForPartition,
    RequestTimedOut,
    MessageTooLarge,
    RecordListTooLarge,
    NotEnoughReplicas,
    NotEnoughReplicasAfterAppend,
}

// Kafka error codes mapping
λ error_from_code(code: i16) -> KafkaError {
    ⌥ code {
        0 => panic!("No error"),
        3 => KafkaError·UnknownTopicOrPartition,
        5 => KafkaError·LeaderNotAvailable,
        6 => KafkaError·NotLeaderForPartition,
        7 => KafkaError·RequestTimedOut,
        10 => KafkaError·MessageTooLarge,
        18 => KafkaError·RecordListTooLarge,
        19 => KafkaError·NotEnoughReplicas,
        20 => KafkaError·NotEnoughReplicasAfterAppend,
        _ => KafkaError·BrokerError { code, message: "Unknown error" },
    }
}
```

---

## 4. Internal Implementation

### 4.1 Connection Management

```sigil
/// Internal TCP connection with Kafka framing
Σ internal·Connection {
    stream: TcpStream,
    buffer: [u8],
    api_versions: Map<i16, (i16, i16)>,  // api_key -> (min, max)
}

impl internal·Connection {
    /// Establish connection to broker
    async λ connect(addr: String) -> Result<Connection, KafkaError> {
        ≔ stream = TcpStream·connect(addr)|await?;
        ≔ conn = Connection {
            stream,
            buffer: [0; 65536],
            api_versions: Map·new(),
        };
        // Negotiate API versions
        conn.negotiate_versions()|await?;
        Ok(conn)
    }

    /// Send request and receive response
    async λ send_request(self, api_key: i16, api_version: i16, body: [u8]) -> Result<[u8], KafkaError> {
        ≔ correlation_id = self.next_correlation_id();

        // Build request with header
        ≔ request = internal·Codec·encode_request(api_key, api_version, correlation_id, body);

        // Send length-prefixed request
        ≔ length = request.len() as i32;
        self.stream|write_all(length|to_be_bytes)|await?;
        self.stream|write_all(request)|await?;
        self.stream|flush|await?;

        // Read response length
        ≔ len_buf = [0u8; 4];
        self.stream|read_exact(len_buf)|await?;
        ≔ response_len = i32·from_be_bytes(len_buf);

        // Read response body
        ≔ response = [0u8; response_len as usize];
        self.stream|read_exact(response)|await?;

        // Verify correlation ID
        ≔ resp_correlation = i32·from_be_bytes(response[0..4]);
        if resp_correlation != correlation_id {
            return Err(KafkaError·InvalidResponse);
        }

        Ok(response[4..])
    }

    /// Negotiate API versions with broker
    async λ negotiate_versions(self) -> Result<(), KafkaError> {
        // ApiVersions request (API key 18)
        ≔ response = self.send_request(18, 0, [])|await?;
        self.api_versions = internal·Codec·decode_api_versions(response)?;
        Ok(())
    }
}
```

### 4.2 Binary Codec

```sigil
/// Binary encoding/decoding for Kafka protocol
mod internal·Codec {
    // === Encoding ===

    λ encode_int16(value: i16) -> [u8; 2] {
        value|to_be_bytes
    }

    λ encode_int32(value: i32) -> [u8; 4] {
        value|to_be_bytes
    }

    λ encode_int64(value: i64) -> [u8; 8] {
        value|to_be_bytes
    }

    λ encode_string(s: String) -> [u8] {
        ≔ bytes = s|as_bytes;
        ≔ len = encode_int16(bytes.len() as i16);
        [..len, ..bytes]
    }

    λ encode_nullable_string(s: Option<String>) -> [u8] {
        ⌥ s {
            Some(str) => encode_string(str),
            None => encode_int16(-1),
        }
    }

    λ encode_bytes(data: [u8]) -> [u8] {
        ≔ len = encode_int32(data.len() as i32);
        [..len, ..data]
    }

    λ encode_array<T>(items: [T], encode_fn: λ(T) -> [u8]) -> [u8] {
        ≔ count = encode_int32(items.len() as i32);
        ≔ encoded = items|flat_map(encode_fn)|collect;
        [..count, ..encoded]
    }

    λ encode_varint(value: i32) -> [u8] {
        // Zig-zag encoding for signed integers
        ≔ encoded = (value << 1) ^ (value >> 31);
        encode_uvarint(encoded as u32)
    }

    λ encode_uvarint(mut value: u32) -> [u8] {
        ≔ result = [];
        ∞ {
            ≔ byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                result|push(byte);
                ⊗;
            } else {
                result|push(byte | 0x80);
            }
        }
        result
    }

    // === Decoding ===

    λ decode_int16(data: [u8], offset: &mut usize) -> i16 {
        ≔ bytes = [data[*offset], data[*offset + 1]];
        *offset += 2;
        i16·from_be_bytes(bytes)
    }

    λ decode_int32(data: [u8], offset: &mut usize) -> i32 {
        ≔ bytes = [data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]];
        *offset += 4;
        i32·from_be_bytes(bytes)
    }

    λ decode_string(data: [u8], offset: &mut usize) -> String {
        ≔ len = decode_int16(data, offset);
        if len < 0 {
            return "";
        }
        ≔ bytes = data[*offset..*offset + len as usize];
        *offset += len as usize;
        String·from_utf8(bytes)|unwrap_or("")
    }

    // === Request Encoding ===

    λ encode_request(api_key: i16, api_version: i16, correlation_id: i32, body: [u8]) -> [u8] {
        ≔ header = [
            ..encode_int16(api_key),
            ..encode_int16(api_version),
            ..encode_int32(correlation_id),
            ..encode_nullable_string(Some("sigil-kafka")),
        ];
        [..header, ..body]
    }

    // === Produce Request (API Key 0) ===

    λ encode_produce_request(
        acks: i16,
        timeout_ms: i32,
        topic_data: [(String, [(i32, [u8])])]  // [(topic, [(partition, records)])]
    ) -> [u8] {
        ≔ transactional_id = encode_nullable_string(None);
        ≔ acks_bytes = encode_int16(acks);
        ≔ timeout_bytes = encode_int32(timeout_ms);

        ≔ topics = encode_array(topic_data, λ((topic, partitions)) {
            ≔ topic_name = encode_string(topic);
            ≔ partition_data = encode_array(partitions, λ((partition, records)) {
                ≔ partition_bytes = encode_int32(partition);
                ≔ record_set = encode_bytes(records);
                [..partition_bytes, ..record_set]
            });
            [..topic_name, ..partition_data]
        });

        [..transactional_id, ..acks_bytes, ..timeout_bytes, ..topics]
    }

    // === Record Batch Encoding ===

    λ encode_record_batch(records: [Record]) -> [u8] {
        // RecordBatch format (v2):
        // baseOffset: int64
        // batchLength: int32
        // partitionLeaderEpoch: int32
        // magic: int8 (2 for v2)
        // crc: int32
        // attributes: int16
        // lastOffsetDelta: int32
        // firstTimestamp: int64
        // maxTimestamp: int64
        // producerId: int64
        // producerEpoch: int16
        // baseSequence: int32
        // records: [Record]

        ≔ now = timestamp_ms();
        ≔ encoded_records = records|enumerate|map(λ(i, r) {
            encode_record(i as i32, r, now)
        })|collect;

        ≔ records_bytes = encoded_records|flatten|collect;

        // Build batch header
        ≔ base_offset = encode_int64(0);
        ≔ partition_leader_epoch = encode_int32(-1);
        ≔ magic = [2u8];  // v2
        ≔ attributes = encode_int16(0);  // no compression for now
        ≔ last_offset_delta = encode_int32((records.len() - 1) as i32);
        ≔ first_timestamp = encode_int64(now);
        ≔ max_timestamp = encode_int64(now);
        ≔ producer_id = encode_int64(-1);
        ≔ producer_epoch = encode_int16(-1);
        ≔ base_sequence = encode_int32(-1);
        ≔ record_count = encode_int32(records.len() as i32);

        ≔ batch_body = [
            ..partition_leader_epoch,
            ..magic,
            // CRC placeholder - will be computed
            ..encode_int32(0),
            ..attributes,
            ..last_offset_delta,
            ..first_timestamp,
            ..max_timestamp,
            ..producer_id,
            ..producer_epoch,
            ..base_sequence,
            ..record_count,
            ..records_bytes,
        ];

        // Compute CRC32-C of batch body (after magic)
        ≔ crc = crc32c(batch_body[5..]);
        batch_body[5..9] = crc|to_be_bytes;

        ≔ batch_length = encode_int32(batch_body.len() as i32);

        [..base_offset, ..batch_length, ..batch_body]
    }

    λ encode_record(offset_delta: i32, record: Record, base_timestamp: i64) -> [u8] {
        ≔ timestamp_delta = record.timestamp|unwrap_or(base_timestamp) - base_timestamp;

        ≔ attributes = [0u8];
        ≔ timestamp_delta_bytes = encode_varint(timestamp_delta as i32);
        ≔ offset_delta_bytes = encode_varint(offset_delta);

        ≔ key_bytes = ⌥ record.key {
            Some(k) => [..encode_varint(k.len() as i32), ..k],
            None => encode_varint(-1),
        };

        ≔ value_bytes = [..encode_varint(record.value.len() as i32), ..record.value];

        ≔ headers_bytes = encode_varint(record.headers.len() as i32);
        ≔ header_data = record.headers|flat_map(λ(k, v) {
            ≔ key = [..encode_varint(k.len() as i32), ..k|as_bytes];
            ≔ val = [..encode_varint(v.len() as i32), ..v];
            [..key, ..val]
        })|collect;

        ≔ body = [
            ..attributes,
            ..timestamp_delta_bytes,
            ..offset_delta_bytes,
            ..key_bytes,
            ..value_bytes,
            ..headers_bytes,
            ..header_data,
        ];

        ≔ length = encode_varint(body.len() as i32);
        [..length, ..body]
    }
}
```

---

## 5. Implementation Phases

### Phase 1: Foundation (P0)
1. Binary codec for primitive types
2. TCP connection with length-prefix framing
3. ApiVersions request/response
4. Metadata request/response

### Phase 2: Producer (P0)
1. Record encoding (v2 format)
2. RecordBatch encoding
3. Produce request/response
4. Basic error handling

### Phase 3: Consumer (P1)
1. Fetch request/response
2. Offset tracking
3. Poll loop implementation

### Phase 4: Hardening (P2)
1. Connection pooling
2. Retry logic
3. Compression support
4. Proper CRC32-C implementation

### Phase 5: Edge Cases (P2 - Future)

The following edge cases are deferred for future hardening work:

| Edge Case | Priority | Description |
|-----------|----------|-------------|
| Empty topic | P2 | Poll returns empty array when no messages available |
| Multiple message batch | P2 | Produce and consume multiple messages in single batch |
| Large messages | P2 | Messages exceeding default buffer size (64KB) |
| Connection timeout | P2 | Graceful handling of broker connection failures |
| Broker failover | P2 | Reconnection logic when broker becomes unavailable |
| Partition reassignment | P2 | Handle partition leadership changes mid-stream |
| Invalid topic | P2 | Graceful error handling for non-existent topics |
| Consumer group coordination | P2 | JoinGroup, SyncGroup, Heartbeat for group consumption |

---

## 6. Test Cases

### 6.1 Unit Tests

```sigil
// Test binary encoding
λ test_encode_int16() {
    assert_eq(Codec·encode_int16(256), [0x01, 0x00]);
    assert_eq(Codec·encode_int16(-1), [0xFF, 0xFF]);
}

λ test_encode_string() {
    assert_eq(Codec·encode_string("hello"), [0x00, 0x05, 'h', 'e', 'l', 'l', 'o']);
}

λ test_encode_varint() {
    assert_eq(Codec·encode_varint(0), [0x00]);
    assert_eq(Codec·encode_varint(1), [0x02]);
    assert_eq(Codec·encode_varint(-1), [0x01]);
    assert_eq(Codec·encode_varint(300), [0xD8, 0x04]);
}
```

### 6.2 Integration Tests

```sigil
// Requires: ./infrastructure/infra.sh up

async λ test_producer_send() {
    ≔ config = ProducerConfig {
        bootstrap_servers: ["localhost:9092"],
        acks: Acks·Leader,
        compression: Compression·None,
        client_id: "sigil-test",
        request_timeout_ms: 30000,
    };

    ≔ producer = Producer·connect(config)|await!;

    ≔ record = Record·new("test-topic", "hello kafka"|as_bytes);
    ≔ metadata = producer.send(record)|await!;

    assert(metadata.offset >= 0);
    println("Sent to partition {metadata.partition} at offset {metadata.offset}");

    producer.close()|await!;
}

async λ test_consumer_poll() {
    ≔ config = ConsumerConfig {
        bootstrap_servers: ["localhost:9092"],
        group_id: None,
        auto_offset_reset: OffsetReset·Earliest,
        client_id: "sigil-test",
        fetch_max_bytes: 1048576,
        fetch_max_wait_ms: 5000,
    };

    ≔ consumer = Consumer·connect(config)|await!;
    consumer.subscribe(["test-topic"])?;

    ≔ records = consumer.poll(5000)|await!;

    for record in records {
        println("Received: {record.value|as_string}");
    }

    consumer.close()|await!;
}
```

### 6.3 Acceptance Criteria

| Test | Criteria |
|------|----------|
| Connection | Successfully connect to broker, negotiate API versions |
| Metadata | Discover topics and partitions |
| Produce | Send message, receive valid offset |
| Fetch | Retrieve previously sent message |
| Error handling | Graceful handling of broker errors |
| Timeout | Respect configured timeouts |

---

## 7. Security Considerations

### 7.1 Current Scope (Plaintext)

Initial implementation supports plaintext connections only. This is suitable for:
- Local development
- Trusted internal networks
- Testing environments

### 7.2 Future: SASL Authentication

```sigil
// Future API for SASL
Σ SaslConfig {
    mechanism: SaslMechanism,
    username: String,
    password: String,
}

ε SaslMechanism {
    Plain,
    ScramSha256,
    ScramSha512,
}
```

### 7.3 Future: TLS

```sigil
// Future API for TLS
Σ TlsConfig {
    ca_cert: Option<String>,
    client_cert: Option<String>,
    client_key: Option<String>,
}
```

---

## 8. Performance Considerations

### 8.1 Batching

- Default batch size: 16KB
- Linger time: 0ms (immediate send)
- Future: configurable batching

### 8.2 Connection Pooling

- Single connection per broker (initial)
- Future: connection pool with configurable size

### 8.3 Buffer Management

- Pre-allocated 64KB receive buffer
- Future: configurable buffer sizes

---

## Appendix A: Kafka API Reference

### A.1 API Keys Used

| Key | Name | Version Range | Notes |
|-----|------|---------------|-------|
| 0 | Produce | 0-9 | Use v3+ for record batches |
| 1 | Fetch | 0-13 | Use v4+ for record batches |
| 3 | Metadata | 0-12 | Topic discovery |
| 18 | ApiVersions | 0-3 | Version negotiation |

### A.2 Error Codes Reference

| Code | Name | Retriable |
|------|------|-----------|
| 0 | NONE | - |
| 3 | UNKNOWN_TOPIC_OR_PARTITION | Yes |
| 5 | LEADER_NOT_AVAILABLE | Yes |
| 6 | NOT_LEADER_FOR_PARTITION | Yes |
| 7 | REQUEST_TIMED_OUT | Yes |
| 10 | MESSAGE_TOO_LARGE | No |
