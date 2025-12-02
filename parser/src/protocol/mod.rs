//! Sigil Protocol Support
//!
//! This module provides comprehensive protocol support for the Sigil standard library,
//! enabling programs to communicate across network boundaries using various protocols.
//!
//! ## Supported Protocols
//!
//! - **gRPC**: High-performance RPC framework with Protocol Buffers
//! - **HTTP**: HTTP/1.1 and HTTP/2 client with streaming support
//! - **WebSocket**: Bidirectional real-time communication
//! - **GraphQL**: Flexible query language client
//! - **Kafka**: Event streaming and message queue
//! - **AMQP**: Message queue protocol (RabbitMQ compatible)
//!
//! ## Philosophy
//!
//! All network data is treated as "reported" (~) evidence by default,
//! requiring explicit validation to promote to "known" (!) status.
//! This ensures programs explicitly handle the uncertainty inherent
//! in distributed systems.

#[cfg(feature = "protocol-core")]
pub mod common;

#[cfg(feature = "http-client")]
pub mod http;

#[cfg(feature = "grpc")]
pub mod grpc;

#[cfg(feature = "websocket")]
pub mod websocket;

#[cfg(feature = "kafka")]
pub mod kafka;

#[cfg(feature = "amqp")]
pub mod amqp;

#[cfg(feature = "graphql")]
pub mod graphql;

// Re-export common types when protocol-core is enabled
#[cfg(feature = "protocol-core")]
pub use common::{
    Uri, Headers, StatusCode, Timeout, RetryConfig, BackoffStrategy,
    ProtocolError, ProtocolResult,
};
