# GPU & Transformer Visualizations

Interactive visualizations for understanding GPU architecture and Transformer attention mechanisms, with AI-powered custom visualization creation. Built with React, TypeScript, and Tailwind CSS.

## Overview

This project provides three types of comprehensive, interactive visualizations:

1. **GPU Architecture Visualization** - Explore NVIDIA GPU Streaming Multiprocessor (SM) components
2. **Transformer Attention Mechanism** - Step-by-step breakdown of self-attention computation
3. **Custom AI-Generated Visualizations** - Create your own visualizations using natural language prompts

## Features

### 🖥️ GPU Architecture Visualization

Interactive demonstration of GPU Streaming Multiprocessor components:

- **Scheduler** - Visualize warp scheduling and parallel execution with adjustable memory pressure
- **CUDA Cores** - See general-purpose ALUs in action with animated activity patterns
- **Tensor Cores** - Understand matrix multiply-accumulate operations (A×B+C) with 16×16 tile visualization
- **Load/Store Units** - Follow the memory hierarchy path from registers to global VRAM
- **Register File** - Explore per-thread register usage and occupancy impacts
- **Shared Memory / L1 Cache** - Compare coalesced vs. strided access patterns with bank conflict visualization

**Interactive Controls:**
- Click any component to see its detailed demo
- Adjust memory pressure to see how the scheduler handles stalled warps
- Toggle between coalesced and strided memory access patterns

### 🧠 Transformer Attention Mechanism

Complete walkthrough of scaled dot-product attention:

**Step-by-step visualization:**
1. Input Embeddings - Token representations
2. Linear Projections - Query (Q), Key (K), Value (V) transformations
3. Attention Scores - Q·K^T computation
4. Scaling - Divide by √d_k
5. Softmax - Convert scores to probability distribution
6. Weighted Sum - Attention × V for final output

**Interactive Controls:**
- **Query Token Selector** - Choose which token to focus on
- **Dimension Slider** - Adjust d_model from 2 to 8 (real models use 512/768/1024)
- **Temperature Control** - Make attention sharper (lower) or more uniform (higher)

**Real-time Updates:**
- All Q, K, V matrices displayed for every token
- Live attention weight calculations
- Visual highlighting of active computations
- Bar charts showing attention distribution

### ✨ Custom AI-Generated Visualizations

Create your own interactive visualizations using natural language:

**How it works:**
1. Click the **"+ Custom"** button in the navigation
2. Describe your visualization in plain English
3. AI generates a complete React/TypeScript component
4. Review the code and save it
5. Your visualization appears instantly in the navigation (no page reload needed!)

**Example prompts:**
- "Create a visualization showing how binary search works with an animated array"
- "Visualize a bubble sort algorithm with color-coded comparisons"
- "Show how a binary tree traversal works with step-by-step highlighting"
- "Create an interactive stack data structure with push and pop operations"

**Powered by:**
- Ollama Gemma 3 for code generation
- Automatic component validation
- Dynamic loading of new visualizations (no page reload needed!)
- Full TypeScript, Framer Motion, and Tailwind CSS support

## Tech Stack

- **React 19** - UI framework with modern hooks
- **TypeScript** - Type safety throughout
- **Vite** - Lightning-fast build tool & dev server
- **Tailwind CSS v4** - Utility-first styling with `@tailwindcss/postcss`
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful icon library
- **Express** - Backend API server
- **Ollama** - Local AI model inference (Gemma 3)

## Installation

### Prerequisites

1. **Node.js** (v18 or higher)
2. **Ollama** - For AI-powered custom visualizations

```bash
# Install Ollama (macOS/Linux)
# Visit: https://ollama.ai/download

# Pull the Gemma 3 model
ollama pull gemma3:latest
```

### Setup

```bash
# Clone or navigate to the project directory
cd gpu-transformer-viz

# Install dependencies
npm install

# Start both frontend and backend servers
npm run dev
```

The app will be available at `http://localhost:5173/`
The backend API runs on `http://localhost:3001/`

### Running Individual Servers

```bash
# Frontend only
npm run dev:frontend

# Backend only
npm run dev:backend
```

## Project Structure

```
gpu-transformer-viz/
├── src/
│   ├── components/
│   │   ├── ui/
│   │   │   ├── card.tsx            # Reusable card component
│   │   │   └── dialog.tsx          # Dialog component
│   │   ├── CustomDialog.tsx        # AI generation dialog
│   │   └── ErrorBoundary.tsx       # Error handling wrapper
│   ├── customized/                 # Custom AI-generated visualizations
│   │   ├── .gitkeep
│   │   └── ExampleViz.tsx          # Example custom visualization
│   ├── utils/
│   │   └── customVizLoader.ts      # Dynamic component loader
│   ├── App.tsx                     # Main app with navigation
│   ├── GPUArchitectureAnimation.tsx # GPU visualization
│   ├── AttentionVisualizer.tsx     # Attention visualization
│   ├── index.css                   # Tailwind CSS imports
│   └── main.tsx                    # App entry point
├── server.js                       # Express backend for Ollama
├── tailwind.config.js              # Tailwind configuration
├── postcss.config.js               # PostCSS with Tailwind plugin
├── vite.config.ts                  # Vite configuration
├── CUSTOM_VIZ_GUIDE.md            # Detailed custom viz documentation
└── package.json                    # Dependencies & scripts
```

## Available Scripts

```bash
# Development
npm run dev              # Start both frontend and backend servers
npm run dev:frontend     # Start Vite dev server only
npm run dev:backend      # Start Express backend only

# Production
npm run build           # Build for production
npm run preview         # Preview production build

# Code Quality
npm run lint            # Run ESLint
```

## Creating Custom Visualizations

### Quick Start

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Launch the app**:
   ```bash
   npm run dev
   ```

3. **Create a visualization**:
   - Click **"+ Custom"** in the navigation
   - Enter your prompt (e.g., "visualize quicksort with color-coded partitions")
   - Click **"Generate Visualization"**
   - Review the generated code
   - Click **"Save Visualization"**
   - Your viz appears in the navigation!

### Advanced Usage

See `CUSTOM_VIZ_GUIDE.md` for:
- Detailed API documentation
- Code format specifications
- Troubleshooting guide
- Best practices and tips
- Example prompts and templates

### File Naming

File names are automatically generated from your prompt by:
- Extracting meaningful keywords (filtering out "create", "show", "visualize", etc.)
- Converting to PascalCase format
- Adding "Viz" suffix

Examples: `LocalitySensitiveHashingViz.tsx`, `BinarySearchAlgorithmViz.tsx`, `BubbleSortViz.tsx`

### Custom Visualization Format

Each custom visualization is a TypeScript React component:

```typescript
import { useState } from 'react';
import { motion } from 'framer-motion';
import { IconName } from 'lucide-react';

export const metadata = {
  name: "Your Viz Name",  // Displayed in navigation
  icon: "IconName"        // Lucide React icon
};

export default function YourVisualization() {
  // Your component code with Tailwind CSS & Framer Motion
  return <div>...</div>;
}
```

### Available in Custom Visualizations

- **React 19** with all hooks (useState, useEffect, useMemo, etc.)
- **TypeScript** for type safety
- **Framer Motion** for animations
- **Lucide React** for 1000+ icons
- **Tailwind CSS** for styling
- **Card components** from `../components/ui/card`

**Note:** For UI controls (sliders, buttons, inputs), use native HTML elements styled with Tailwind CSS. See CUSTOM_VIZ_GUIDE.md for examples.

## Learning Resources

### GPU Architecture
- Understand warp scheduling and latency hiding
- Learn how CUDA cores execute instructions
- Explore memory hierarchy and coalescing
- See bank conflicts in shared memory
- Visualize Tensor Core matrix operations

### Transformer Attention
- Follow the exact math behind attention
- See how Q, K, V projections work
- Understand scaling and its importance
- Observe softmax normalization
- Watch weighted value aggregation

### AI-Generated Visualizations
- Algorithm visualizations (sorting, searching)
- Data structures (trees, graphs, stacks, queues)
- Mathematical concepts (Fourier transforms, gradients)
- Computer graphics (ray tracing, bezier curves)
- Neural network architectures

## Technical Notes

### Tailwind CSS v4
This project uses Tailwind CSS v4, which requires:
- `@tailwindcss/postcss` plugin (not the old `tailwindcss` PostCSS plugin)
- `@import "tailwindcss"` in CSS (not `@tailwind` directives)

### Custom Colors
Theme colors for shadcn/ui components are defined in `tailwind.config.js`:
- `card` - Card background and foreground
- `muted` - Muted text colors

### Hot Module Replacement (HMR)
- Vite automatically reloads when you edit existing components
- New custom visualizations appear instantly without page reload
- Backend API tracks all available visualizations dynamically

## API Endpoints

The backend server provides these endpoints:

- `POST /api/generate` - Generate visualization from prompt
- `POST /api/save` - Save generated code to file
- `GET /api/list` - List all custom visualizations
- `GET /api/health` - Health check

See `CUSTOM_VIZ_GUIDE.md` for detailed API documentation.

## Troubleshooting

### Custom Visualizations Not Working?

1. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

2. **Verify Gemma 3 is installed**:
   ```bash
   ollama list
   ```

3. **Check backend is running**:
   - Should see "Backend server running on http://localhost:3001" in terminal

4. **Check browser console** for errors

### Common Issues

- **"Failed to generate"** - Ollama not running or model not installed
- **Validation warnings** - Code may still work; regenerate if issues persist
- **Component not appearing** - Refresh page or check `/src/customized/` folder
- **TypeScript errors** - Try regenerating with a clearer, more specific prompt

## Browser Support

Modern browsers with ES6+ support:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Contributing

This is an educational visualization project. Feel free to:
- Report issues or bugs
- Suggest new visualizations or features
- Improve animations or interactivity
- Add more GPU components or attention variants
- Share interesting custom visualizations

## Security

- AI-generated code runs in React Error Boundaries
- Basic validation checks for required exports
- Files saved to sandboxed `/src/customized/` folder
- No eval() or Function() constructors used

## License

MIT

## Acknowledgments

- GPU architecture concepts based on NVIDIA CUDA programming guides
- Transformer attention mechanism from "Attention Is All You Need" paper
- AI code generation powered by Ollama and Gemma 3
- Built with modern React and Vite ecosystem
