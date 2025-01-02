---
title: Lightweight Embeddings
emoji: 🌍
colorFrom: green
colorTo: green
sdk: docker
app_file: app.py
---

# 🌍 LightweightEmbeddings: Multilingual, Fast, and Unlimited

**LightweightEmbeddings** is a fast, free, and unlimited API service for multilingual embeddings and reranking, with support for both text and images and guaranteed uptime.

## ✨ Key Features

- **Free and Unlimited**: A completely free API service with no limits on usage, making it accessible for everyone.
- **Multilingual Support**: Seamlessly process text in over 100+ languages for global applications.
- **Text and Image Embeddings**: Generate high-quality embeddings from text or image-text pairs using state-of-the-art models.
- **Reranking Support**: Includes powerful reranking capabilities for both text and image inputs.
- **Optimized for Speed**: Built with lightweight transformer models and efficient backends for rapid inference, even on low-resource systems.
- **Flexible Model Support**: Use a range of transformer models tailored to diverse use cases:
  - Text models: `multilingual-e5-small`, `paraphrase-multilingual-MiniLM-L12-v2`, `bge-m3`
  - Image model: `google/siglip-base-patch16-256-multilingual`
- **Production-Ready**: Easily deploy anywhere with Docker for hassle-free setup.
- **Interactive Playground**: Test embeddings and reranking directly via a **Gradio-powered interface** alongside detailed REST API documentation.

## 🚀 Use Cases

- **Search and Ranking**: Generate embeddings for advanced similarity-based ranking in search engines.
- **Recommendation Systems**: Use embeddings for personalized recommendations based on user input or preferences.
- **Multimodal Applications**: Combine text and image embeddings to power tasks like product catalog indexing, content moderation, or multimodal retrieval.
- **Language Understanding**: Enable semantic text analysis, summarization, or classification in multiple languages.

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/lh0x00/lightweight-embeddings.git
cd lightweight-embeddings
```

### 2. Build and Run with Docker
Make sure Docker is installed and running on your machine.
```bash
docker build -t lightweight-embeddings .
docker run -p 7860:7860 lightweight-embeddings
```

The API will now be accessible at `http://localhost:7860`.

## 📖 API Overview

### Endpoints
- **`/v1/embeddings`**: Generate text or image embeddings using the model of your choice.
- **`/v1/rank`**: Rank candidate inputs based on similarity to a query.

### Interactive Docs
- Visit the [Swagger UI](http://localhost:7860/docs) for detailed, interactive documentation.
- Explore additional resources with [ReDoc](http://localhost:7860/redoc).

## 🔬 Playground

### Embeddings Playground
- Test text and image embedding generation in the browser with a user-friendly **Gradio interface**.
- Simply visit `http://localhost:7860` after starting the server to access the playground.

## 🌐 Resources

- **Documentation**: [Explore full documentation](https://lamhieu-lightweight-embeddings.hf.space/docs)
- **Hugging Face Space**: [Try the live demo](https://huggingface.co/spaces/lamhieu/lightweight-embeddings)
- **GitHub Repository**: [View source code](https://github.com/lh0x00/lightweight-embeddings)

## 💡 Why LightweightEmbeddings?

1. **Performance-Oriented**: Delivers rapid results without compromising on quality, ideal for real-world deployment.
2. **Highly Adaptable**: Works in diverse environments, from cloud clusters to local devices.
3. **Developer-Friendly**: Intuitive API design with robust documentation and an integrated playground for experimentation.

## 👥 Contributors

- **lamhieu** – Creator and Maintainer ([GitHub](https://github.com/lh0x00))

Contributions are welcome! Check out the [contribution guidelines](https://github.com/lh0x00/lightweight-embeddings/blob/main/CONTRIBUTING.md).

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/lh0x00/lightweight-embeddings/blob/main/LICENSE) file for details.
