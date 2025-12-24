//! Embeddings API endpoint for OpenRouter.
//!
//! This module provides access to the OpenRouter embeddings API, which allows
//! generating vector embeddings from text using various embedding models.
//!
//! # Example
//! ```rust,no_run
//! use openrouter_api::OpenRouterClient;
//! use openrouter_api::types::embeddings::{EmbeddingRequest, EmbeddingInput};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = OpenRouterClient::from_env()?;
//!
//! // Simple single embedding
//! let response = client.embeddings()?
//!     .create(EmbeddingRequest::new(
//!         "openai/text-embedding-3-small",
//!         "Hello, world!"
//!     ))
//!     .await?;
//!
//! println!("Embedding dimensions: {:?}", response.dimensions());
//!
//! // Batch embeddings
//! let batch_response = client.embeddings()?
//!     .create(EmbeddingRequest::new(
//!         "openai/text-embedding-3-small",
//!         vec!["First text", "Second text", "Third text"]
//!     ))
//!     .await?;
//!
//! println!("Generated {} embeddings", batch_response.len());
//! # Ok(())
//! # }
//! ```

use crate::error::{Error, Result};
use crate::types::embeddings::{EmbeddingRequest, EmbeddingResponse};
use crate::types::models::ModelsResponse;
use crate::utils::retry::operations::EMBEDDINGS;
use crate::utils::{retry::execute_with_retry_builder, retry::handle_response_json, validation};
use reqwest::Client;

/// API endpoint for text embeddings.
pub struct EmbeddingsApi {
    pub client: Client,
    pub config: crate::client::ApiConfig,
}

impl EmbeddingsApi {
    /// Creates a new EmbeddingsApi with the given reqwest client and configuration.
    pub fn new(client: Client, config: &crate::client::ClientConfig) -> Result<Self> {
        Ok(Self {
            client,
            config: config.to_api_config()?,
        })
    }

    /// Creates embeddings for the given input text(s).
    ///
    /// # Arguments
    /// * `request` - The embedding request containing the model and input text(s).
    ///
    /// # Returns
    /// An `EmbeddingResponse` containing the generated embeddings.
    ///
    /// # Example
    /// ```rust,no_run
    /// use openrouter_api::OpenRouterClient;
    /// use openrouter_api::types::embeddings::EmbeddingRequest;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenRouterClient::from_env()?;
    ///
    /// let response = client.embeddings()?
    ///     .create(EmbeddingRequest::new(
    ///         "openai/text-embedding-3-small",
    ///         "The quick brown fox jumps over the lazy dog"
    ///     ))
    ///     .await?;
    ///
    /// if let Some(embedding) = response.first_embedding() {
    ///     println!("Embedding has {} dimensions", embedding.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // Validate the request
        validation::validate_embedding_request(&request)?;

        // Build the URL for the embeddings endpoint
        let url = self
            .config
            .base_url
            .join("embeddings")
            .map_err(|e| Error::ApiError {
                code: 400,
                message: format!("Invalid URL for embeddings endpoint: {e}"),
                metadata: None,
            })?;

        // Use pre-built headers from config
        let headers = self.config.headers.clone();

        // Execute request with retry logic
        let response = execute_with_retry_builder(&self.config.retry_config, EMBEDDINGS, || {
            self.client
                .post(url.clone())
                .headers(headers.clone())
                .json(&request)
        })
        .await?;

        // Handle response with consistent error parsing
        handle_response_json::<EmbeddingResponse>(response, EMBEDDINGS).await
    }

    /// Simple helper to create a single embedding.
    ///
    /// # Arguments
    /// * `model` - The model to use for generating the embedding.
    /// * `text` - The text to embed.
    ///
    /// # Returns
    /// The embedding vector as a `Vec<f32>`.
    ///
    /// # Example
    /// ```rust,no_run
    /// use openrouter_api::OpenRouterClient;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenRouterClient::from_env()?;
    ///
    /// let embedding = client.embeddings()?
    ///     .embed("openai/text-embedding-3-small", "Hello, world!")
    ///     .await?;
    ///
    /// println!("Embedding: {:?}", &embedding[..5]); // First 5 values
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest::new(model, text);
        let response = self.create(request).await?;

        response
            .first_embedding()
            .cloned()
            .ok_or_else(|| Error::ApiError {
                code: 500,
                message: "No embedding returned in response".to_string(),
                metadata: None,
            })
    }

    /// Creates embeddings for multiple texts in a single request.
    ///
    /// # Arguments
    /// * `model` - The model to use for generating embeddings.
    /// * `texts` - The texts to embed.
    ///
    /// # Returns
    /// A vector of embedding vectors.
    ///
    /// # Example
    /// ```rust,no_run
    /// use openrouter_api::OpenRouterClient;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenRouterClient::from_env()?;
    ///
    /// let embeddings = client.embeddings()?
    ///     .embed_batch(
    ///         "openai/text-embedding-3-small",
    ///         &["First document", "Second document", "Third document"]
    ///     )
    ///     .await?;
    ///
    /// println!("Generated {} embeddings", embeddings.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed_batch(&self, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let input: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let request = EmbeddingRequest::new(model, input);
        let response = self.create(request).await?;

        response
            .all_embeddings()
            .map(|embeddings| embeddings.into_iter().cloned().collect())
            .ok_or_else(|| Error::ApiError {
                code: 500,
                message: "Failed to extract embeddings from response".to_string(),
                metadata: None,
            })
    }

    /// Lists available embedding models.
    ///
    /// # Returns
    /// A `ModelsResponse` containing all available embedding models.
    ///
    /// # Example
    /// ```rust,no_run
    /// use openrouter_api::OpenRouterClient;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenRouterClient::from_env()?;
    ///
    /// let models = client.embeddings()?.list_models().await?;
    ///
    /// for model in &models.data {
    ///     println!("{}: {}", model.id, model.name);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_models(&self) -> Result<ModelsResponse> {
        // Build the URL for the embeddings models endpoint
        let url = self
            .config
            .base_url
            .join("embeddings/models")
            .map_err(|e| Error::ApiError {
                code: 400,
                message: format!("Invalid URL for embeddings models endpoint: {e}"),
                metadata: None,
            })?;

        // Use pre-built headers from config
        let headers = self.config.headers.clone();

        // Execute request with retry logic
        let response =
            execute_with_retry_builder(&self.config.retry_config, EMBEDDINGS_MODELS, || {
                self.client.get(url.clone()).headers(headers.clone())
            })
            .await?;

        // Handle response with consistent error parsing
        handle_response_json::<ModelsResponse>(response, EMBEDDINGS_MODELS).await
    }
}

/// Operation name for embeddings requests
const EMBEDDINGS_MODELS: &str = "embeddings_models";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::embeddings::EmbeddingInput;

    #[test]
    fn test_embedding_request_creation() {
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", "test input");
        assert_eq!(request.model, "openai/text-embedding-3-small");
        match request.input {
            EmbeddingInput::Single(s) => assert_eq!(s, "test input"),
            _ => panic!("Expected Single input"),
        }
    }

    #[test]
    fn test_batch_embedding_request() {
        let texts: Vec<String> = vec!["first".to_string(), "second".to_string()];
        let request = EmbeddingRequest::new("openai/text-embedding-3-small", texts);
        match request.input {
            EmbeddingInput::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], "first");
                assert_eq!(v[1], "second");
            }
            _ => panic!("Expected Multiple input"),
        }
    }
}
