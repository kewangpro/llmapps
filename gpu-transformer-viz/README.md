# GPU & Transformer Visualizations

Interactive visualizations for understanding GPU architecture and Transformer attention mechanisms, built with React, TypeScript, and Tailwind CSS.

## Overview

This project provides two comprehensive, interactive visualizations:

1. **GPU Architecture Visualization** - Explore NVIDIA GPU Streaming Multiprocessor (SM) components
2. **Transformer Attention Mechanism** - Step-by-step breakdown of self-attention computation

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

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Tailwind CSS v4** - Styling with `@tailwindcss/postcss`
- **Framer Motion** - Smooth animations
- **Lucide React** - Icon library

## Installation

```bash
# Clone or navigate to the project directory
cd gpu-transformer-viz

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173/`

## Project Structure

```
gpu-transformer-viz/
├── src/
│   ├── components/
│   │   └── ui/
│   │       └── card.tsx            # Reusable UI card component
│   ├── App.tsx                     # Main app with navigation
│   ├── GPUArchitectureAnimation.tsx # GPU viz component
│   ├── AttentionVisualizer.tsx     # Attention viz component
│   ├── index.css                   # Tailwind CSS imports
│   ├── main.tsx                    # App entry point
│   └── App.css                     # Unused Vite boilerplate
├── tailwind.config.js              # Tailwind configuration
├── postcss.config.js               # PostCSS with Tailwind plugin
├── vite.config.ts                  # Vite configuration
└── package.json                    # Dependencies
```

## Available Scripts

```bash
# Development
npm run dev          # Start dev server with HMR

# Production
npm run build        # Build for production
npm run preview      # Preview production build

# Code Quality
npm run lint         # Run ESLint
```

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

## Technical Notes

### Tailwind CSS v4
This project uses Tailwind CSS v4, which requires:
- `@tailwindcss/postcss` plugin (not the old `tailwindcss` PostCSS plugin)
- `@import "tailwindcss"` in CSS (not `@tailwind` directives)

### Custom Colors
Theme colors for shadcn/ui components are defined in `tailwind.config.js`:
- `card` - Card background and foreground
- `muted` - Muted text colors

## Browser Support

Modern browsers with ES6+ support:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Contributing

This is an educational visualization project. Feel free to:
- Report issues or bugs
- Suggest new visualizations
- Improve animations or interactivity
- Add more GPU components or attention variants

## License

MIT

## Acknowledgments

- GPU architecture concepts based on NVIDIA CUDA programming guides
- Transformer attention mechanism from "Attention Is All You Need" paper
- Built with modern React and Vite ecosystem
