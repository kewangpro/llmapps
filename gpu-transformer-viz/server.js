import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// System prompt for Ollama to generate visualization components
const SYSTEM_PROMPT = `You are an expert React/TypeScript developer creating interactive visualizations.

Generate a complete, self-contained React component that:
1. Uses React hooks (useState, useEffect, useMemo, etc.)
2. Uses Tailwind CSS for styling (utility classes)
3. Uses Framer Motion for animations (import from 'framer-motion')
4. Uses Lucide React for icons (import from 'lucide-react')
5. ONLY uses Card components from '../components/ui/card' if needed
6. Is fully typed with TypeScript
7. Is educational and interactive with controls (sliders, buttons, toggles)

AVAILABLE UI COMPONENTS:
- Card, CardHeader, CardTitle, CardDescription, CardContent from '../components/ui/card'
- That's it! Do NOT import slider, button, select, or any other UI components.

FOR CONTROLS, BUILD THEM DIRECTLY WITH HTML + TAILWIND:
- Sliders: Use <input type="range" className="..." />
- Buttons: Use <button className="px-4 py-2 bg-blue-600 text-white rounded-lg..." />
- Select: Use <select className="..." />
- Checkboxes: Use <input type="checkbox" className="..." />
- All styled with Tailwind CSS utility classes

REQUIRED STRUCTURE:
\`\`\`typescript
import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';

export const metadata = {
  name: "Example Viz",
  icon: "Sparkles"
};

export default function ExampleViz() {
  const [value, setValue] = useState(0);

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Title</h2>

        {/* Simple animation example */}
        <motion.div
          animate={{ scale: 1, opacity: 1 }}
          initial={{ scale: 0.8, opacity: 0 }}
          className="w-32 h-32 bg-blue-500 rounded-lg"
        />

        {/* Interactive control example */}
        <input
          type="range"
          value={value}
          onChange={(e) => setValue(Number(e.target.value))}
          className="w-full"
        />
      </div>
    </div>
  );
}
\`\`\`

IMPORTANT:
- Only output the complete TypeScript code, no explanations
- Only import what you actually use - remove unused imports
- Include necessary imports: useState, useEffect from 'react', motion from 'framer-motion', icons from 'lucide-react'
- Make it visually appealing with colors and animations
- Add interactive controls using native HTML elements styled with Tailwind
- DO NOT import components that don't exist (no slider, button, select components)
- Use Tailwind classes for all styling
- Component should be self-contained and not depend on external state
- Choose a short, descriptive name for metadata.name (max 20 characters)

LUCIDE ICON RULES (CRITICAL):
- Icon names do NOT have "Icon" prefix: use "Sparkles" NOT "IconSparkles"
- Icon names are PascalCase: "ArrowUpDown" NOT "Sort" or "IconSort"
- Valid sorting/algorithm icons: ArrowUpDown, ArrowDownUp, ArrowUpNarrowWide, BarChart, Activity
- Valid data structure icons: Database, Box, Layers, Grid, Network, Binary
- Valid animation icons: Play, Pause, RotateCw, Zap, Sparkles, TrendingUp
- Valid general icons: Rocket, Brain, Code, Terminal, Hash, TrendingUp
- Import example: import { ArrowUpDown, Play } from 'lucide-react'
- In metadata, use the exact same name: icon: "ArrowUpDown"
- NEVER use: IconSort, IconBrain, IconSparkles, Sort - these don't exist!

FRAMER MOTION RULES (CRITICAL):
- Use EITHER animate/initial OR variants, NEVER both together
- Simple animation: <motion.div animate={{ scale: 1 }} initial={{ scale: 0 }}>
- With variants: <motion.div variants={myVariants} animate="visible" initial="hidden">
- Keep it simple - don't over-complicate animations
- Correct: animate={{ x: 100, opacity: 1 }}
- Wrong: animate={{ scale: 1, transition: { duration: 1 }}} initial={{ scale: 0.8 }} variants={{...}}
- Transition goes inside animate: animate={{ scale: 1, transition: { duration: 0.5 } }}`;

// Call Ollama API
async function callOllama(prompt) {
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3:latest',
        prompt: `${SYSTEM_PROMPT}\n\nUser request: ${prompt}`,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    throw new Error(`Failed to call Ollama: ${error.message}`);
  }
}

// Extract code from markdown code blocks if present
function extractCode(text) {
  const codeBlockRegex = /```(?:typescript|tsx|ts)?\n([\s\S]*?)```/;
  const match = text.match(codeBlockRegex);
  return match ? match[1].trim() : text.trim();
}

// Basic validation of generated code
function validateCode(code) {
  const errors = [];

  // Check for React import
  if (!code.includes('import React') && !code.includes('import {') && !code.includes('from \'react\'')) {
    errors.push('Missing React import');
  }

  // Check for metadata export
  if (!code.includes('export const metadata')) {
    errors.push('Missing metadata export');
  }

  // Check for default export
  if (!code.includes('export default')) {
    errors.push('Missing default export');
  }

  // Check for JSX return
  if (!code.includes('return')) {
    errors.push('Component must return JSX');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}

// Generate component name from prompt
function generateComponentName(prompt) {
  // Common stop words to filter out
  const stopWords = new Set([
    'a', 'an', 'the', 'create', 'make', 'show', 'display', 'visualize', 'visualization',
    'demonstrate', 'how', 'to', 'of', 'for', 'with', 'that', 'is', 'are', 'be',
    'new', 'interactive', 'animated', 'simple', 'basic', 'and', 'or', 'but',
    'in', 'on', 'at', 'by', 'from', 'about', 'works', 'work', 'working'
  ]);

  // Extract meaningful words
  const words = prompt
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '') // Keep hyphens for compound words
    .split(/\s+/)
    .filter(w => w.length > 2) // Filter short words
    .filter(w => !stopWords.has(w)) // Filter stop words
    .slice(0, 4); // Max 4 meaningful words

  // If no meaningful words found, use a generic name with timestamp
  if (words.length === 0) {
    return 'CustomViz' + Date.now();
  }

  // Convert to PascalCase
  const name = words
    .map(word => word.split('-').map(part =>
      part.charAt(0).toUpperCase() + part.slice(1)
    ).join(''))
    .join('');

  return name + 'Viz';
}

// API endpoint to generate visualization
app.post('/api/generate', async (req, res) => {
  try {
    const { prompt } = req.body;

    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    console.log(`Generating visualization for prompt: "${prompt}"`);

    // Call Ollama
    const rawResponse = await callOllama(prompt);

    // Extract code from markdown if present
    const code = extractCode(rawResponse);

    // Validate code
    const validation = validateCode(code);

    // Generate component name
    const componentName = generateComponentName(prompt);

    res.json({
      code,
      componentName,
      validation,
      rawResponse
    });

  } catch (error) {
    console.error('Error generating visualization:', error);
    res.status(500).json({
      error: error.message || 'Failed to generate visualization'
    });
  }
});

// API endpoint to save visualization
app.post('/api/save', async (req, res) => {
  try {
    const { code, fileName } = req.body;

    if (!code || !fileName) {
      return res.status(400).json({ error: 'Code and fileName are required' });
    }

    // Ensure fileName is safe
    const safeFileName = fileName.replace(/[^a-zA-Z0-9]/g, '') + '.tsx';
    const customizedDir = path.join(__dirname, 'src', 'customized');
    const filePath = path.join(customizedDir, safeFileName);

    // Create customized directory if it doesn't exist
    await fs.mkdir(customizedDir, { recursive: true });

    // Write file
    await fs.writeFile(filePath, code, 'utf-8');

    console.log(`Saved visualization to: ${filePath}`);

    res.json({
      success: true,
      fileName: safeFileName,
      path: filePath
    });

  } catch (error) {
    console.error('Error saving visualization:', error);
    res.status(500).json({
      error: error.message || 'Failed to save visualization'
    });
  }
});

// API endpoint to list custom visualizations
app.get('/api/list', async (req, res) => {
  try {
    const customizedDir = path.join(__dirname, 'src', 'customized');

    // Create directory if it doesn't exist
    await fs.mkdir(customizedDir, { recursive: true });

    const files = await fs.readdir(customizedDir);
    const tsxFiles = files.filter(f => f.endsWith('.tsx'));

    res.json({ files: tsxFiles });

  } catch (error) {
    console.error('Error listing visualizations:', error);
    res.status(500).json({
      error: error.message || 'Failed to list visualizations'
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

app.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`📝 Make sure Ollama is running: ollama serve`);
  console.log(`📦 Make sure gemma3 model is installed: ollama pull gemma3:latest`);
});
