//! Validation utilities for embeddings API requests

use crate::error::{Error, Result};
use crate::types::embeddings::{EmbeddingInput, EmbeddingRequest};

use super::common::{validate_model_id, validate_non_empty_string, validate_numeric_min};

/// Maximum number of inputs in a single batch request
const MAX_BATCH_SIZE: usize = 2048;

/// Maximum length of a single input text (in characters)
const MAX_INPUT_LENGTH: usize = 100_000;

/// Maximum number of tokens in a single input (approximate)
const MAX_INPUT_TOKENS: usize = 8191;

/// Validates an embedding request
pub fn validate_embedding_request(request: &EmbeddingRequest) -> Result<()> {
    // Validate model ID
    validate_model_id(&request.model)?;

    // Validate input
    validate_embedding_input(&request.input)?;

    // Validate dimensions if specified
    if let Some(dimensions) = request.dimensions {
        validate_numeric_min(dimensions, "dimensions", 1)?;
        // Most models cap at 3072 dimensions, but some go higher
        if dimensions > 10_000 {
            return Err(Error::ConfigError(format!(
                "Dimensions {} exceeds maximum allowed (10000)",
                dimensions
            )));
        }
    }

    // Validate input_type if specified
    if let Some(ref input_type) = request.input_type {
        validate_non_empty_string(input_type, "input_type")?;
        // Common input types
        let valid_types = [
            "query",
            "document",
            "search_query",
            "search_document",
            "classification",
            "clustering",
        ];
        // Just warn if not a common type, don't error since providers may support others
        if !valid_types.contains(&input_type.as_str()) {
            // Allow any non-empty string, just validate it's not empty
        }
    }

    // Validate user if specified
    if let Some(ref user) = request.user {
        validate_non_empty_string(user, "user")?;
    }

    Ok(())
}

/// Validates the embedding input
fn validate_embedding_input(input: &EmbeddingInput) -> Result<()> {
    match input {
        EmbeddingInput::Single(text) => {
            validate_single_input(text, 0)?;
        }
        EmbeddingInput::Multiple(texts) => {
            if texts.is_empty() {
                return Err(Error::ConfigError(
                    "Embedding input cannot be empty".to_string(),
                ));
            }
            if texts.len() > MAX_BATCH_SIZE {
                return Err(Error::ConfigError(format!(
                    "Batch size {} exceeds maximum allowed ({})",
                    texts.len(),
                    MAX_BATCH_SIZE
                )));
            }
            for (index, text) in texts.iter().enumerate() {
                validate_single_input(text, index)?;
            }
        }
        EmbeddingInput::Tokens(token_arrays) => {
            if token_arrays.is_empty() {
                return Err(Error::ConfigError(
                    "Embedding input cannot be empty".to_string(),
                ));
            }
            if token_arrays.len() > MAX_BATCH_SIZE {
                return Err(Error::ConfigError(format!(
                    "Batch size {} exceeds maximum allowed ({})",
                    token_arrays.len(),
                    MAX_BATCH_SIZE
                )));
            }
            for (index, tokens) in token_arrays.iter().enumerate() {
                if tokens.is_empty() {
                    return Err(Error::ConfigError(format!(
                        "Token array at index {} cannot be empty",
                        index
                    )));
                }
                if tokens.len() > MAX_INPUT_TOKENS {
                    return Err(Error::ConfigError(format!(
                        "Token array at index {} has {} tokens, exceeds maximum ({})",
                        index,
                        tokens.len(),
                        MAX_INPUT_TOKENS
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Validates a single text input
fn validate_single_input(text: &str, index: usize) -> Result<()> {
    if text.is_empty() {
        return Err(Error::ConfigError(format!(
            "Input text at index {} cannot be empty",
            index
        )));
    }
    if text.len() > MAX_INPUT_LENGTH {
        return Err(Error::ConfigError(format!(
            "Input text at index {} has {} characters, exceeds maximum ({})",
            index,
            text.len(),
            MAX_INPUT_LENGTH
        )));
    }
    Ok(())
}

/// Estimates the number of tokens in a text (rough approximation)
/// Uses a simple heuristic of ~4 characters per token for English text
#[allow(dead_code)]
pub fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 characters per token for English
    // This is a simplified estimate; actual tokenization varies by model
    (text.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::embeddings::EncodingFormat;

    #[test]
    fn test_validate_single_input() {
        // Valid single input
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "Hello, world!");
        assert!(validate_embedding_request(&request).is_ok());
    }

    #[test]
    fn test_validate_batch_input() {
        // Valid batch input
        let request = EmbeddingRequest::new(
            "openai/text-embedding-3-small",
            vec!["First text".to_string(), "Second text".to_string()],
        );
        assert!(validate_embedding_request(&request).is_ok());
    }

    #[test]
    fn test_validate_empty_input() {
        // Empty single input
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "");
        assert!(validate_embedding_request(&request).is_err());
    }

    #[test]
    fn test_validate_empty_batch() {
        // Empty batch
        let request =
            EmbeddingRequest::new("openai/text-embedding-3-small", Vec::<String>::new());
        assert!(validate_embedding_request(&request).is_err());
    }

    #[test]
    fn test_validate_invalid_model() {
        // Invalid model ID (missing slash)
        let request = EmbeddingRequest::new("invalid-model", "Hello");
        assert!(validate_embedding_request(&request).is_err());
    }

    #[test]
    fn test_validate_dimensions() {
        // Valid dimensions
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "Hello")
            .with_dimensions(256);
        assert!(validate_embedding_request(&request).is_ok());

        // Invalid dimensions (too high)
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "Hello")
            .with_dimensions(20000);
        assert!(validate_embedding_request(&request).is_err());
    }

    #[test]
    fn test_validate_with_options() {
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "Hello, world!")
            .with_encoding_format(EncodingFormat::Float)
            .with_dimensions(512)
            .with_input_type("query")
            .with_user("user-123");

        assert!(validate_embedding_request(&request).is_ok());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello"), 2); // 5 chars -> ~2 tokens
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars -> ~3 tokens
        assert_eq!(estimate_tokens(""), 0); // empty -> 0 tokens
    }

    #[test]
    fn test_validate_input_too_long() {
        // Create a very long input
        let long_text = "a".repeat(MAX_INPUT_LENGTH + 1);
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", long_text);
        assert!(validate_embedding_request(&request).is_err());
    }

    #[test]
    fn test_validate_batch_too_large() {
        // Create a batch that's too large
        let large_batch: Vec<String> = (0..MAX_BATCH_SIZE + 1)
            .map(|i| format!("Text {}", i))
            .collect();
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", large_batch);
        assert!(validate_embedding_request(&request).is_err());
    }
}
