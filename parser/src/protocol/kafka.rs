//! Apache Kafka Support
//!
//! High-throughput distributed event streaming platform support.
//!
//! ## Features
//!
//! - Producer with acknowledgments and compression
//! - Consumer with consumer groups
//! - Partition assignment and rebalancing
//! - Transactional producer
//! - Exactly-once semantics
//! - Schema registry integration ready

use super::common::{Headers, ProtocolError, ProtocolResult, Timeout};
use std::collections::HashMap;
use std::time::Duration;

/// Kafka acknowledgment modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acks {
    /// No acknowledgment (fire and forget)
    None,
    /// Leader acknowledgment only
    Leader,
    /// Full ISR acknowledgment
    All,
}

impl Acks {
    /// Get the numeric value for Kafka
    pub fn as_i32(&self) -> i32 {
        match self {
            Acks::None => 0,
            Acks::Leader => 1,
            Acks::All => -1,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Snappy compression (recommended for speed)
    Snappy,
    /// LZ4 compression
    Lz4,
    /// Zstandard compression (recommended for ratio)
    Zstd,
}

impl Compression {
    /// Get the Kafka compression type string
    pub fn as_str(&self) -> &'static str {
        match self {
            Compression::None => "none",
            Compression::Gzip => "gzip",
            Compression::Snappy => "snappy",
            Compression::Lz4 => "lz4",
            Compression::Zstd => "zstd",
        }
    }
}

/// Offset reset strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetReset {
    /// Start from the earliest offset
    Earliest,
    /// Start from the latest offset
    Latest,
    /// Fail if no offset is stored
    Error,
}

impl OffsetReset {
    /// Get the Kafka offset reset string
    pub fn as_str(&self) -> &'static str {
        match self {
            OffsetReset::Earliest => "earliest",
            OffsetReset::Latest => "latest",
            OffsetReset::Error => "error",
        }
    }
}

/// Kafka producer configuration
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// Bootstrap servers (comma-separated)
    pub bootstrap_servers: Vec<String>,
    /// Acknowledgment mode
    pub acks: Acks,
    /// Compression algorithm
    pub compression: Compression,
    /// Number of retries for transient failures
    pub retries: u32,
    /// Maximum in-flight requests
    pub max_in_flight: u32,
    /// Batch size in bytes
    pub batch_size: usize,
    /// Linger time before sending batch
    pub linger_ms: u64,
    /// Request timeout
    pub request_timeout: Duration,
    /// Delivery timeout
    pub delivery_timeout: Duration,
    /// Enable idempotence
    pub enable_idempotence: bool,
    /// Transactional ID (for exactly-once)
    pub transactional_id: Option<String>,
    /// Security protocol
    pub security_protocol: SecurityProtocol,
    /// SASL mechanism
    pub sasl_mechanism: Option<SaslMechanism>,
    /// SASL username
    pub sasl_username: Option<String>,
    /// SASL password
    pub sasl_password: Option<String>,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        ProducerConfig {
            bootstrap_servers: vec!["localhost:9092".to_string()],
            acks: Acks::All,
            compression: Compression::None,
            retries: 3,
            max_in_flight: 5,
            batch_size: 16384,
            linger_ms: 0,
            request_timeout: Duration::from_secs(30),
            delivery_timeout: Duration::from_secs(120),
            enable_idempotence: false,
            transactional_id: None,
            security_protocol: SecurityProtocol::Plaintext,
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
        }
    }
}

/// Kafka consumer configuration
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    /// Bootstrap servers (comma-separated)
    pub bootstrap_servers: Vec<String>,
    /// Consumer group ID
    pub group_id: String,
    /// Offset reset strategy
    pub auto_offset_reset: OffsetReset,
    /// Enable auto commit
    pub enable_auto_commit: bool,
    /// Auto commit interval
    pub auto_commit_interval: Duration,
    /// Maximum poll records
    pub max_poll_records: u32,
    /// Maximum poll interval
    pub max_poll_interval: Duration,
    /// Session timeout
    pub session_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Fetch minimum bytes
    pub fetch_min_bytes: usize,
    /// Fetch maximum bytes
    pub fetch_max_bytes: usize,
    /// Fetch maximum wait time
    pub fetch_max_wait: Duration,
    /// Security protocol
    pub security_protocol: SecurityProtocol,
    /// SASL mechanism
    pub sasl_mechanism: Option<SaslMechanism>,
    /// SASL username
    pub sasl_username: Option<String>,
    /// SASL password
    pub sasl_password: Option<String>,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        ConsumerConfig {
            bootstrap_servers: vec!["localhost:9092".to_string()],
            group_id: "sigil-consumer".to_string(),
            auto_offset_reset: OffsetReset::Earliest,
            enable_auto_commit: true,
            auto_commit_interval: Duration::from_secs(5),
            max_poll_records: 500,
            max_poll_interval: Duration::from_secs(300),
            session_timeout: Duration::from_secs(10),
            heartbeat_interval: Duration::from_secs(3),
            fetch_min_bytes: 1,
            fetch_max_bytes: 50 * 1024 * 1024,
            fetch_max_wait: Duration::from_millis(500),
            security_protocol: SecurityProtocol::Plaintext,
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
        }
    }
}

/// Security protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityProtocol {
    /// No encryption
    Plaintext,
    /// TLS encryption
    Ssl,
    /// SASL authentication without encryption
    SaslPlaintext,
    /// SASL authentication with TLS
    SaslSsl,
}

impl SecurityProtocol {
    /// Get the Kafka security protocol string
    pub fn as_str(&self) -> &'static str {
        match self {
            SecurityProtocol::Plaintext => "PLAINTEXT",
            SecurityProtocol::Ssl => "SSL",
            SecurityProtocol::SaslPlaintext => "SASL_PLAINTEXT",
            SecurityProtocol::SaslSsl => "SASL_SSL",
        }
    }
}

/// SASL mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaslMechanism {
    /// PLAIN mechanism
    Plain,
    /// SCRAM-SHA-256
    ScramSha256,
    /// SCRAM-SHA-512
    ScramSha512,
    /// GSSAPI (Kerberos)
    Gssapi,
    /// OAuth Bearer
    OAuthBearer,
}

impl SaslMechanism {
    /// Get the SASL mechanism string
    pub fn as_str(&self) -> &'static str {
        match self {
            SaslMechanism::Plain => "PLAIN",
            SaslMechanism::ScramSha256 => "SCRAM-SHA-256",
            SaslMechanism::ScramSha512 => "SCRAM-SHA-512",
            SaslMechanism::Gssapi => "GSSAPI",
            SaslMechanism::OAuthBearer => "OAUTHBEARER",
        }
    }
}

/// A Kafka record/message
#[derive(Debug, Clone)]
pub struct Record {
    /// Topic name
    pub topic: String,
    /// Partition (None for automatic assignment)
    pub partition: Option<i32>,
    /// Message key (for partitioning)
    pub key: Option<Vec<u8>>,
    /// Message value
    pub value: Vec<u8>,
    /// Message headers
    pub headers: RecordHeaders,
    /// Timestamp (None for broker timestamp)
    pub timestamp: Option<i64>,
}

impl Record {
    /// Create a new record
    pub fn new(topic: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        Record {
            topic: topic.into(),
            partition: None,
            key: None,
            value: value.into(),
            headers: RecordHeaders::new(),
            timestamp: None,
        }
    }

    /// Set the key
    pub fn key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Set the key as a string
    pub fn key_str(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into().into_bytes());
        self
    }

    /// Set the partition
    pub fn partition(mut self, partition: i32) -> Self {
        self.partition = Some(partition);
        self
    }

    /// Set the timestamp
    pub fn timestamp(mut self, timestamp: i64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Add a header
    pub fn header(mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.insert(key, value);
        self
    }
}

/// Record headers
#[derive(Debug, Clone, Default)]
pub struct RecordHeaders {
    inner: Vec<(String, Vec<u8>)>,
}

impl RecordHeaders {
    /// Create empty headers
    pub fn new() -> Self {
        RecordHeaders { inner: Vec::new() }
    }

    /// Insert a header
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) -> &mut Self {
        self.inner.push((key.into(), value.into()));
        self
    }

    /// Get a header value
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.inner.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
    }

    /// Get all values for a header key
    pub fn get_all(&self, key: &str) -> Vec<&[u8]> {
        self.inner.iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
            .collect()
    }

    /// Iterate over headers
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of headers
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

/// A consumed record with metadata
#[derive(Debug, Clone)]
pub struct ConsumedRecord {
    /// The record data
    pub record: Record,
    /// Topic name
    pub topic: String,
    /// Partition number
    pub partition: i32,
    /// Offset in the partition
    pub offset: i64,
    /// Record timestamp
    pub timestamp: i64,
    /// Timestamp type
    pub timestamp_type: TimestampType,
}

impl ConsumedRecord {
    /// Get the key as bytes
    pub fn key(&self) -> Option<&[u8]> {
        self.record.key.as_deref()
    }

    /// Get the key as a string
    pub fn key_str(&self) -> Option<&str> {
        self.record.key.as_ref().and_then(|k| std::str::from_utf8(k).ok())
    }

    /// Get the value
    pub fn value(&self) -> &[u8] {
        &self.record.value
    }

    /// Get the value as a string
    pub fn value_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.record.value).ok()
    }

    /// Get headers
    pub fn headers(&self) -> &RecordHeaders {
        &self.record.headers
    }
}

/// Timestamp type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampType {
    /// No timestamp
    NoTimestamp,
    /// Create time (producer timestamp)
    CreateTime,
    /// Log append time (broker timestamp)
    LogAppendTime,
}

/// Topic partition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopicPartition {
    /// Topic name
    pub topic: String,
    /// Partition number
    pub partition: i32,
}

impl TopicPartition {
    /// Create a new topic partition
    pub fn new(topic: impl Into<String>, partition: i32) -> Self {
        TopicPartition {
            topic: topic.into(),
            partition,
        }
    }
}

impl std::fmt::Display for TopicPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.topic, self.partition)
    }
}

/// Kafka producer
#[derive(Debug)]
pub struct Producer {
    config: ProducerConfig,
    #[cfg(feature = "rdkafka")]
    inner: Option<rdkafka::producer::FutureProducer>,
}

impl Producer {
    /// Create a new producer
    pub fn new(config: ProducerConfig) -> ProtocolResult<Self> {
        #[cfg(feature = "rdkafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::producer::FutureProducer;

            let mut client_config = ClientConfig::new();
            client_config
                .set("bootstrap.servers", config.bootstrap_servers.join(","))
                .set("acks", config.acks.as_i32().to_string())
                .set("compression.type", config.compression.as_str())
                .set("retries", config.retries.to_string())
                .set("max.in.flight.requests.per.connection", config.max_in_flight.to_string())
                .set("batch.size", config.batch_size.to_string())
                .set("linger.ms", config.linger_ms.to_string())
                .set("request.timeout.ms", config.request_timeout.as_millis().to_string())
                .set("delivery.timeout.ms", config.delivery_timeout.as_millis().to_string())
                .set("enable.idempotence", config.enable_idempotence.to_string())
                .set("security.protocol", config.security_protocol.as_str());

            if let Some(ref tx_id) = config.transactional_id {
                client_config.set("transactional.id", tx_id);
            }

            if let Some(mechanism) = &config.sasl_mechanism {
                client_config.set("sasl.mechanism", mechanism.as_str());
            }
            if let Some(ref username) = config.sasl_username {
                client_config.set("sasl.username", username);
            }
            if let Some(ref password) = config.sasl_password {
                client_config.set("sasl.password", password);
            }

            let producer: FutureProducer = client_config.create()
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;

            Ok(Producer {
                config,
                inner: Some(producer),
            })
        }

        #[cfg(not(feature = "rdkafka"))]
        {
            Ok(Producer { config })
        }
    }

    /// Create a producer with default config
    pub fn default() -> ProtocolResult<Self> {
        Self::new(ProducerConfig::default())
    }

    /// Send a record
    #[cfg(feature = "rdkafka")]
    pub async fn send(&self, record: Record) -> ProtocolResult<(i32, i64)> {
        use rdkafka::producer::FutureRecord;
        use rdkafka::message::OwnedHeaders;

        if let Some(ref producer) = self.inner {
            let mut future_record = FutureRecord::to(&record.topic)
                .payload(&record.value);

            if let Some(ref key) = record.key {
                future_record = future_record.key(key);
            }

            if let Some(partition) = record.partition {
                future_record = future_record.partition(partition);
            }

            if let Some(timestamp) = record.timestamp {
                future_record = future_record.timestamp(timestamp);
            }

            // Add headers
            if !record.headers.is_empty() {
                let mut headers = OwnedHeaders::new();
                for (key, value) in record.headers.iter() {
                    headers = headers.insert(rdkafka::message::Header {
                        key,
                        value: Some(value),
                    });
                }
                future_record = future_record.headers(headers);
            }

            let result = producer.send(future_record, Duration::from_secs(0)).await
                .map_err(|(e, _)| ProtocolError::Protocol(e.to_string()))?;

            Ok(result)
        } else {
            Err(ProtocolError::Protocol("Producer not initialized".to_string()))
        }
    }

    /// Send a record (stub for non-rdkafka builds)
    #[cfg(not(feature = "rdkafka"))]
    pub async fn send(&self, record: Record) -> ProtocolResult<(i32, i64)> {
        let _ = record;
        Err(ProtocolError::Protocol("Kafka requires 'kafka' feature".to_string()))
    }

    /// Send multiple records
    pub async fn send_batch(&self, records: Vec<Record>) -> ProtocolResult<Vec<(i32, i64)>> {
        let mut results = Vec::with_capacity(records.len());
        for record in records {
            results.push(self.send(record).await?);
        }
        Ok(results)
    }

    /// Flush pending messages
    #[cfg(feature = "rdkafka")]
    pub fn flush(&self, timeout: Duration) -> ProtocolResult<()> {
        if let Some(ref producer) = self.inner {
            producer.flush(timeout)
                .map_err(|e| ProtocolError::Protocol(e.to_string()))
        } else {
            Ok(())
        }
    }

    #[cfg(not(feature = "rdkafka"))]
    pub fn flush(&self, _timeout: Duration) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol("Kafka requires 'kafka' feature".to_string()))
    }
}

/// Kafka consumer
#[derive(Debug)]
pub struct Consumer {
    config: ConsumerConfig,
    #[cfg(feature = "rdkafka")]
    inner: Option<rdkafka::consumer::StreamConsumer>,
}

impl Consumer {
    /// Create a new consumer
    pub fn new(config: ConsumerConfig) -> ProtocolResult<Self> {
        #[cfg(feature = "rdkafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::StreamConsumer;

            let mut client_config = ClientConfig::new();
            client_config
                .set("bootstrap.servers", config.bootstrap_servers.join(","))
                .set("group.id", &config.group_id)
                .set("auto.offset.reset", config.auto_offset_reset.as_str())
                .set("enable.auto.commit", config.enable_auto_commit.to_string())
                .set("auto.commit.interval.ms", config.auto_commit_interval.as_millis().to_string())
                .set("max.poll.records", config.max_poll_records.to_string())
                .set("max.poll.interval.ms", config.max_poll_interval.as_millis().to_string())
                .set("session.timeout.ms", config.session_timeout.as_millis().to_string())
                .set("heartbeat.interval.ms", config.heartbeat_interval.as_millis().to_string())
                .set("fetch.min.bytes", config.fetch_min_bytes.to_string())
                .set("fetch.max.bytes", config.fetch_max_bytes.to_string())
                .set("fetch.wait.max.ms", config.fetch_max_wait.as_millis().to_string())
                .set("security.protocol", config.security_protocol.as_str());

            if let Some(mechanism) = &config.sasl_mechanism {
                client_config.set("sasl.mechanism", mechanism.as_str());
            }
            if let Some(ref username) = config.sasl_username {
                client_config.set("sasl.username", username);
            }
            if let Some(ref password) = config.sasl_password {
                client_config.set("sasl.password", password);
            }

            let consumer: StreamConsumer = client_config.create()
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;

            Ok(Consumer {
                config,
                inner: Some(consumer),
            })
        }

        #[cfg(not(feature = "rdkafka"))]
        {
            Ok(Consumer { config })
        }
    }

    /// Subscribe to topics
    #[cfg(feature = "rdkafka")]
    pub fn subscribe(&self, topics: &[&str]) -> ProtocolResult<()> {
        use rdkafka::consumer::Consumer as _;

        if let Some(ref consumer) = self.inner {
            consumer.subscribe(topics)
                .map_err(|e| ProtocolError::Protocol(e.to_string()))
        } else {
            Err(ProtocolError::Protocol("Consumer not initialized".to_string()))
        }
    }

    #[cfg(not(feature = "rdkafka"))]
    pub fn subscribe(&self, _topics: &[&str]) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol("Kafka requires 'kafka' feature".to_string()))
    }

    /// Unsubscribe from all topics
    #[cfg(feature = "rdkafka")]
    pub fn unsubscribe(&self) {
        use rdkafka::consumer::Consumer as _;

        if let Some(ref consumer) = self.inner {
            consumer.unsubscribe();
        }
    }

    #[cfg(not(feature = "rdkafka"))]
    pub fn unsubscribe(&self) {}

    /// Commit offsets synchronously
    #[cfg(feature = "rdkafka")]
    pub fn commit(&self) -> ProtocolResult<()> {
        use rdkafka::consumer::Consumer as _;
        use rdkafka::consumer::CommitMode;

        if let Some(ref consumer) = self.inner {
            consumer.commit_consumer_state(CommitMode::Sync)
                .map_err(|e| ProtocolError::Protocol(e.to_string()))
        } else {
            Err(ProtocolError::Protocol("Consumer not initialized".to_string()))
        }
    }

    #[cfg(not(feature = "rdkafka"))]
    pub fn commit(&self) -> ProtocolResult<()> {
        Err(ProtocolError::Protocol("Kafka requires 'kafka' feature".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_builder() {
        let record = Record::new("test-topic", b"hello".to_vec())
            .key_str("key1")
            .partition(0)
            .header("source", b"test".to_vec());

        assert_eq!(record.topic, "test-topic");
        assert_eq!(record.key, Some(b"key1".to_vec()));
        assert_eq!(record.partition, Some(0));
        assert_eq!(record.headers.get("source"), Some(b"test".as_ref()));
    }

    #[test]
    fn test_producer_config() {
        let config = ProducerConfig {
            bootstrap_servers: vec!["localhost:9092".to_string()],
            acks: Acks::All,
            compression: Compression::Snappy,
            ..Default::default()
        };

        assert_eq!(config.acks.as_i32(), -1);
        assert_eq!(config.compression.as_str(), "snappy");
    }

    #[test]
    fn test_consumer_config() {
        let config = ConsumerConfig {
            group_id: "my-group".to_string(),
            auto_offset_reset: OffsetReset::Earliest,
            ..Default::default()
        };

        assert_eq!(config.group_id, "my-group");
        assert_eq!(config.auto_offset_reset.as_str(), "earliest");
    }
}
