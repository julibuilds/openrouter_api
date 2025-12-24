//! Types for the OpenRouter Embeddings API.
//!
//! This module provides request and response types for generating text embeddings
//! using OpenRouter's unified embeddings API.

use serde::{Deserialize, Serialize};

use crate::types::provider::ProviderPreferences;

/// Input format for embeddings requests.
///
/// Supports multiple input formats as per the OpenRouter API:
/// - Single string
/// - Array of strings (for batch processing)
/// - Array of token arrays (pre-tokenized input)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text string to embed.
    Single(String),
    /// Multiple text strings to embed in a single request.
    Multiple(Vec<String>),
    /// Pre-tokenized input as arrays of token IDs.
    Tokens(Vec<Vec<i32>>),
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        EmbeddingInput::Single(s)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        EmbeddingInput::Single(s.to_string())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        EmbeddingInput::Multiple(v)
    }
}

impl From<Vec<&str>> for EmbeddingInput {
    fn from(v: Vec<&str>) -> Self {
        EmbeddingInput::Multiple(v.into_iter().map(String::from).collect())
    }
}

/// Encoding format for the embedding output.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Return embeddings as an array of floating point numbers (default).
    #[default]
    Float,
    /// Return embeddings as a base64-encoded string.
    Base64,
}

/// Request to generate embeddings.
///
/// # Example
/// ```rust
/// use openrouter_api::types::embeddings::{EmbeddingRequest, EmbeddingInput};
///
/// let request = EmbeddingRequest {
///     model: "openai/text-embedding-3-small".to_string(),
///     input: EmbeddingInput::Single("Hello, world!".to_string()),
///     encoding_format: None,
///     dimensions: None,
///     input_type: None,
///     provider: None,
///     user: None,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// The model to use for generating embeddings.
    /// Example: "openai/text-embedding-3-small"
    pub model: String,

    /// The input text(s) to embed.
    pub input: EmbeddingInput,

    /// The format to return the embeddings in.
    /// Can be "float" (default) or "base64".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,

    /// The number of dimensions for the output embeddings.
    /// Only supported by some models (e.g., text-embedding-3-*).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// The type of input being embedded.
    /// Some models use this to optimize embeddings (e.g., "query" vs "document").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,

    /// Provider routing preferences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,

    /// A unique identifier for the end-user, for abuse detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingRequest {
    /// Creates a new embedding request with the given model and input.
    pub fn new(model: impl Into<String>, input: impl Into<EmbeddingInput>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            encoding_format: None,
            dimensions: None,
            input_type: None,
            provider: None,
            user: None,
        }
    }

    /// Sets the encoding format for the embeddings.
    pub fn with_encoding_format(mut self, format: EncodingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Sets the number of dimensions for the output embeddings.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Sets the input type for optimization.
    pub fn with_input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = Some(input_type.into());
        self
    }

    /// Sets provider routing preferences.
    pub fn with_provider(mut self, provider: ProviderPreferences) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Sets the user identifier for abuse detection.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// The embedding data for a single input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// The object type, always "embedding".
    pub object: String,

    /// The index of this embedding in the request's input array.
    pub index: usize,

    /// The embedding vector.
    /// This will be a Vec<f32> for float encoding, or a base64 string for base64 encoding.
    pub embedding: EmbeddingValue,
}

/// The embedding value, which can be either float array or base64 encoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    /// Embedding as an array of floating point numbers.
    Float(Vec<f32>),
    /// Embedding as a base64-encoded string.
    Base64(String),
}

impl EmbeddingValue {
    /// Returns the embedding as a float vector if available.
    pub fn as_float(&self) -> Option<&Vec<f32>> {
        match self {
            EmbeddingValue::Float(v) => Some(v),
            EmbeddingValue::Base64(_) => None,
        }
    }

    /// Returns the embedding as a base64 string if available.
    pub fn as_base64(&self) -> Option<&str> {
        match self {
            EmbeddingValue::Float(_) => None,
            EmbeddingValue::Base64(s) => Some(s),
        }
    }

    /// Returns the dimensionality of the embedding.
    pub fn dimensions(&self) -> Option<usize> {
        match self {
            EmbeddingValue::Float(v) => Some(v.len()),
            EmbeddingValue::Base64(_) => None, // Can't determine without decoding
        }
    }
}

/// Usage information for an embeddings request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// The number of tokens in the input.
    pub prompt_tokens: u32,

    /// The total number of tokens used.
    pub total_tokens: u32,

    /// The cost of the request, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

/// Response from an embeddings request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The object type, always "list".
    pub object: String,

    /// The list of embedding data.
    pub data: Vec<EmbeddingData>,

    /// The model used for the embeddings.
    pub model: String,

    /// Usage information for this request.
    pub usage: EmbeddingUsage,
}

impl EmbeddingResponse {
    /// Returns the first embedding as a float vector, if available.
    pub fn first_embedding(&self) -> Option<&Vec<f32>> {
        self.data.first().and_then(|d| d.embedding.as_float())
    }

    /// Returns all embeddings as float vectors.
    /// Returns None if any embedding is not in float format.
    pub fn all_embeddings(&self) -> Option<Vec<&Vec<f32>>> {
        self.data
            .iter()
            .map(|d| d.embedding.as_float())
            .collect::<Option<Vec<_>>>()
    }

    /// Returns the total number of embeddings in the response.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the response contains no embeddings.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the dimensionality of the embeddings.
    /// Returns None if no embeddings or if embeddings are base64 encoded.
    pub fn dimensions(&self) -> Option<usize> {
        self.data.first().and_then(|d| d.embedding.dimensions())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_from_string() {
        let input: EmbeddingInput = "hello".into();
        match input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello"),
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_embedding_input_from_vec() {
        let input: EmbeddingInput = vec!["hello", "world"].into();
        match input {
            EmbeddingInput::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], "hello");
                assert_eq!(v[1], "world");
            }
            _ => panic!("Expected Multiple variant"),
        }
    }

    #[test]
    fn test_embedding_request_builder() {
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "test input")
            .with_dimensions(256)
            .with_encoding_format(EncodingFormat::Float)
            .with_input_type("query")
            .with_user("user-123");

        assert_eq!(request.model, "openai/text-embedding-3-small");
        assert_eq!(request.dimensions, Some(256));
        assert_eq!(request.encoding_format, Some(EncodingFormat::Float));
        assert_eq!(request.input_type, Some("query".to_string()));
        assert_eq!(request.user, Some("user-123".to_string()));
    }

    #[test]
    fn test_embedding_request_serialization() {
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "test input");
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("openai/text-embedding-3-small"));
        assert!(json.contains("test input"));
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            ],
            "model": "openai/text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }"#;

        let response: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.model, "openai/text-embedding-3-small");
        assert_eq!(response.usage.prompt_tokens, 5);
        assert_eq!(response.usage.total_tokens, 5);

        let embedding = response.first_embedding().unwrap();
        assert_eq!(embedding.len(), 5);
        assert!((embedding[0] - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_embedding_value_helpers() {
        let float_val = EmbeddingValue::Float(vec![1.0, 2.0, 3.0]);
        assert!(float_val.as_float().is_some());
        assert!(float_val.as_base64().is_none());
        assert_eq!(float_val.dimensions(), Some(3));

        let base64_val = EmbeddingValue::Base64("SGVsbG8=".to_string());
        assert!(base64_val.as_float().is_none());
        assert!(base64_val.as_base64().is_some());
        assert!(base64_val.dimensions().is_none());
    }

    #[test]
    fn test_embedding_response_helpers() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![
                EmbeddingData {
                    object: "embedding".to_string(),
                    index: 0,
                    embedding: EmbeddingValue::Float(vec![1.0, 2.0]),
                },
                EmbeddingData {
                    object: "embedding".to_string(),
                    index: 1,
                    embedding: EmbeddingValue::Float(vec![3.0, 4.0]),
                },
            ],
            model: "test-model".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
                cost: Some(0.001),
            },
        };

        assert_eq!(response.len(), 2);
        assert!(!response.is_empty());
        assert_eq!(response.dimensions(), Some(2));
        assert!(response.all_embeddings().is_some());
        assert_eq!(response.all_embeddings().unwrap().len(), 2);
    }
}
