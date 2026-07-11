# Custom Visualization Feature Guide

## Overview

The Custom Visualization feature allows you to create AI-generated interactive visualizations using natural language prompts. This feature uses Ollama's Gemma 3 model to generate React/TypeScript components that seamlessly integrate with the existing GPU and Transformer visualizations.

### Current Status

✅ **AI System Prompt Updated:**
- Explicit rules about Lucide icon naming (no "Icon" prefix)
- Framer Motion animation guidelines to prevent syntax errors
- List of valid icon names for different categories
- Instructions to only import what's used
- Native HTML controls with Tailwind styling

✅ **Console Logging Added:**
- Browser console shows each step of the loading process
- Backend logs show when files are saved
- Easy to debug issues with emoji indicators

✅ **UI Improvements:**
- Built-in visualizations (GPU, Attention) always visible
- "+ Custom" button always accessible
- Custom visualizations scroll independently in the middle

---

## Prerequisites

### 1. Ollama Installation

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull the Gemma 3 model
ollama pull gemma3:latest

# Start Ollama server (if not already running)
ollama serve
```

### 2. Node.js Dependencies

```bash
npm install
```

---

## Getting Started

### 1. Start the Development Server

Run both the frontend (Vite) and backend (Express) servers concurrently:

```bash
npm run dev
```

This starts:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3001

### 2. Open Browser Console

Open http://localhost:5173 and press **F12** to open DevTools Console. This helps you see what's happening during generation and loading.

### 3. Generate a Test Visualization

Click **"+ Custom"** button and try this prompt:
```
Create a visualization showing how bubble sort works with an animated array
```

### 4. Watch the Logs

**In Browser Console, you should see:**
```
🔄 Visualization created, reloading list...
📥 Fetching updated visualization list...
⚙️ Loading visualizations...
🔍 Fetching file list from backend...
📁 Files found: ["ExampleViz.tsx", "BubbleSortViz.tsx"]
📦 Loading ExampleViz.tsx...
✅ Loaded: Example Viz (icon: Rocket)
📦 Loading BubbleSortViz.tsx...
✅ Loaded: Bubble Sort (icon: ArrowUpDown)
📊 Total visualizations loaded: 2
💾 Setting state with 2 visualizations: ["Example Viz", "Bubble Sort"]
✅ Visualization list reloaded
```

**In Terminal, you should see:**
```
[1] Generating visualization for prompt: "Create a visualization..."
[1] Saved visualization to: /Users/.../BubbleSortViz.tsx
```

### 5. Verify the Visualization Appears

After 1-2 seconds, you should see a new button in the navigation bar with your custom visualization name.

---

## Navigation Layout

The navigation bar is organized to keep important buttons always visible:

**Layout:**
```
[GPU] [Attention]  |  [Custom Viz 1] [Custom Viz 2] ⬅️➡️  |  [+ Custom]
   Fixed                 Scrollable middle              Fixed
```

**Benefits:**
- GPU and Attention buttons always visible
- "+ Custom" button always accessible
- Custom visualizations scroll independently (if you have many)
- Shorter button labels for more space

---

## Example Prompts

### ✅ Good Prompts (Specific and Clear)

**Algorithm Visualizations:**
- "Create a visualization showing how bubble sort works with an animated array"
- "Visualize quicksort algorithm with pivot highlighting"
- "Create a binary search tree visualization with insert and search operations"
- "Show how a hash table handles collisions with chaining"

**Data Structure Visualizations:**
- "Create an interactive stack data structure with push/pop operations"
- "Visualize a linked list with add, remove, and search operations"
- "Show how a priority queue works with animated priority-based sorting"

**Math & Physics Visualizations:**
- "Create a visualization of the Fourier transform with adjustable frequency"
- "Show how gradient descent optimization works in 2D"
- "Visualize sine and cosine waves with phase and amplitude controls"

**Computer Graphics:**
- "Create a ray tracing visualization showing light bounces"
- "Visualize bezier curves with draggable control points"
- "Show how different blending modes work in graphics"

### ❌ Bad Prompts (Too Vague)

- "Make something cool"
- "Algorithm visualization"
- "Show me code"

---

## File Naming

The backend automatically generates descriptive file names from your prompt:

**How it works:**
- Filters out stop words ("create", "show", "visualize", "demonstrate", etc.)
- Extracts meaningful keywords (up to 4 words)
- Handles hyphenated terms (e.g., "Locality-Sensitive" stays together)
- Converts to PascalCase with "Viz" suffix

**Examples:**
| Prompt | Generated Filename |
|--------|-------------------|
| "create a visualization for Locality-Sensitive Hashing" | `LocalitySensitiveHashingViz.tsx` |
| "visualize binary search algorithm" | `BinarySearchAlgorithmViz.tsx` |
| "show bubble sort with animations" | `BubbleSortAnimationsViz.tsx` |
| "interactive stack data structure" | `StackDataStructureViz.tsx` |

**Tip:** Use descriptive, specific terms in your prompt for better file names!

---

## Custom Visualization Format

Each generated visualization follows this structure:

```typescript
import { useState } from 'react';
import { motion } from 'framer-motion';
import { IconName } from 'lucide-react';

// Required metadata export
export const metadata = {
  name: "Your Viz Name",  // Max 20 characters
  icon: "IconName"        // Lucide React icon name
};

// Required default export (name should match filename)
export default function YourVisualizationViz() {
  const [value, setValue] = useState(0);

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="bg-white rounded-lg shadow-lg p-6">
        {/* Your visualization UI */}
      </div>
    </div>
  );
}
```

---

## Available Libraries

All custom visualizations have access to:

- **React 19** - Modern React with hooks (useState, useEffect, useMemo, etc.)
- **TypeScript** - Full type safety
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - 1000+ clean icons
- **Tailwind CSS** - Utility-first styling
- **Card Components** - Card, CardHeader, CardTitle, CardDescription, CardContent from `../components/ui/card`

**IMPORTANT:** Only the Card components are available. For other UI controls (sliders, buttons, inputs), use native HTML elements styled with Tailwind CSS.

### Building Controls with Tailwind

**Sliders:**
```tsx
<input
  type="range"
  min="0"
  max="100"
  value={value}
  onChange={(e) => setValue(Number(e.target.value))}
  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
/>
```

**Buttons:**
```tsx
<button
  onClick={handleClick}
  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
>
  Click Me
</button>
```

**Select Dropdowns:**
```tsx
<select
  value={selected}
  onChange={(e) => setSelected(e.target.value)}
  className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
>
  <option value="opt1">Option 1</option>
  <option value="opt2">Option 2</option>
</select>
```

**Checkboxes/Toggles:**
```tsx
<label className="flex items-center gap-2 cursor-pointer">
  <input
    type="checkbox"
    checked={enabled}
    onChange={(e) => setEnabled(e.target.checked)}
    className="w-4 h-4 text-blue-600 rounded"
  />
  <span className="text-sm text-gray-700">Enable Feature</span>
</label>
```

---

## Valid Lucide Icons by Category

**Sorting/Algorithms:**
- ArrowUpDown, ArrowDownUp, ArrowUpNarrowWide, BarChart, Activity

**Data Structures:**
- Database, Box, Layers, Grid, Network, Binary

**Animation/Action:**
- Play, Pause, RotateCw, Zap, Sparkles, TrendingUp

**General:**
- Rocket, Brain, Code, Terminal, Hash

**Note:** Icon names do NOT have "Icon" prefix. Use "Sparkles" NOT "IconSparkles".

---

## File Structure

```
src/
├── customized/              # Custom visualization folder
│   ├── .gitkeep
│   ├── ExampleViz.tsx       # Example visualization
│   └── YourCustomViz.tsx    # Your generated visualizations
├── components/
│   ├── CustomDialog.tsx     # Generation dialog
│   ├── ErrorBoundary.tsx    # Error handling
│   └── ui/
│       ├── card.tsx
│       └── dialog.tsx
├── utils/
│   └── customVizLoader.ts   # Dynamic component loader
└── App.tsx                  # Main app with navigation
```

---

## API Endpoints

The backend server provides these endpoints:

### `POST /api/generate`
Generate a new visualization from a prompt.

**Request:**
```json
{
  "prompt": "Create a visualization showing..."
}
```

**Response:**
```json
{
  "code": "// Generated TypeScript code",
  "componentName": "YourVizName",
  "validation": {
    "isValid": true,
    "errors": []
  }
}
```

### `POST /api/save`
Save generated code to a file.

### `GET /api/list`
List all custom visualizations.

### `GET /api/health`
Health check endpoint.

---

## Troubleshooting

### Issue: "Failed to generate visualization"

**Symptom:** Error message when clicking "Generate Visualization"

**Cause:** Ollama not running or model not installed

**Solution:**
1. Ensure Ollama is running: `ollama serve`
2. Check Gemma 3 is installed: `ollama list`
3. Verify backend is running on port 3001
4. Check terminal logs for errors

---

### Issue: "IconSort" or "IconXyz" Import Errors

**Symptom:**
```
error TS2305: Module '"lucide-react"' has no exported member 'IconSort'
```

**Cause:** AI generated incorrect icon name with "Icon" prefix

**Common Errors:**
```tsx
❌ import { IconSort } from 'lucide-react';      // Wrong
❌ import { IconSparkles } from 'lucide-react';  // Wrong
✅ import { ArrowUpDown } from 'lucide-react';   // Correct
✅ import { Sparkles } from 'lucide-react';      // Correct
```

**Solution:**
- Delete the broken file from `/src/customized/`
- Try generating again - the updated AI prompt should fix it
- If it persists, manually edit the file and remove "Icon" prefix

---

### Issue: Visualization Doesn't Appear After Save

**Check Browser Console for:**
1. Are files being fetched? Look for `📁 Files found:`
2. Are there import errors? Look for `❌ Failed to load`
3. Is state being updated? Look for `💾 Setting state with X visualizations`

**Check Backend Terminal for:**
1. Was the file saved? Look for `Saved visualization to:`
2. Check the file exists: `ls src/customized/`

**Common Fixes:**
1. Wait 2-3 seconds (there's a 1.5s delay intentionally)
2. Check if the file has TypeScript errors: `npm run build`
3. Look at the generated file for obvious issues
4. Manually refresh the page if needed

---

### Issue: Generated Code Has Errors

**Common Errors from AI:**

1. **Unused imports**
   ```tsx
   ❌ import { motion } from 'framer-motion';  // Never used
   ✅ Remove unused imports
   ```

2. **Wrong TypeScript types**
   ```tsx
   ❌ const [steps, setSteps] = useState<number[]>([]);
   ✅ const [steps, setSteps] = useState<Step[]>([]);
   // Define proper interface first
   ```

3. **Invalid property access**
   ```tsx
   ❌ step.array  // TypeScript error if 'step' is typed as 'number'
   ✅ Ensure types match the data structure
   ```

4. **Wrong component name**
   ```tsx
   ❌ export default function ExampleViz() {
   ✅ export default function YourActualFilename() {
   ```

5. **Invalid Lucide icon props**
   ```tsx
   ❌ <Activity lineWeight={2} size={32} color="gray" />
   ✅ <Activity strokeWidth={2} size={32} className="text-gray-500" />
   ```

6. **Framer Motion conflicts**
   ```tsx
   ❌ <motion.div animate={{...}} initial={{...}} variants={{...}} />
   // Can't use all three together!
   ✅ <motion.div animate={{ scale: 1 }} initial={{ scale: 0 }} />
   // Use EITHER animate/initial OR variants, not both
   ```

**Solution:**
- Delete the file from `/src/customized/`
- Regenerate with a clearer, more specific prompt
- Or manually edit the file to fix the issues

---

### Issue: "Validation warnings"

**Solution:**
- The code may still work despite warnings
- If the visualization doesn't render, try regenerating with a more specific prompt
- Check browser console for detailed error messages

---

## Manual Verification

```bash
# Check what files backend sees
curl http://localhost:3001/api/list

# Check for TypeScript errors
npm run build

# List generated files
ls -la src/customized/
```

---

## Success Criteria

When everything works correctly:

✅ New visualization appears in navigation within 2 seconds
✅ No TypeScript compilation errors
✅ Can click the button and see the visualization
✅ Console logs show successful loading
✅ File saved with correct naming (e.g., `BubbleSortViz.tsx`)

---

## Dynamic Loading

Custom visualizations are loaded dynamically without requiring page reloads:

1. When you save a new visualization, it appears instantly in the navigation
2. The backend API tracks all available visualizations in real-time
3. Dynamic imports with cache-busting ensure fresh module loading
4. Edits to existing visualizations are picked up by Vite's HMR

**How it works:**
- Frontend queries backend for file list via `/api/list`
- Files are dynamically imported using `import(/* @vite-ignore */ path)`
- Timestamp parameter prevents stale cached modules
- New visualizations appear after ~1.5 second delay

---

## Manual Editing

You can manually create or edit visualizations:

1. Create a `.tsx` file in `/src/customized/`
2. Follow the required format (metadata + default export)
3. Save the file
4. Vite HMR will automatically pick it up

---

## Best Practices

1. **Clear Prompts:** Be specific about what you want to visualize
2. **Interactive Controls:** Request sliders, buttons, or toggles for better UX
3. **Animations:** Ask for Framer Motion animations to make it engaging
4. **Color Coding:** Use Tailwind colors to highlight different states
5. **Educational Value:** Include explanations or tooltips
6. **Step-by-Step:** For algorithms, request step-through controls

---

## Example: Bubble Sort Visualization

**Prompt:**
```
Create a visualization showing how bubble sort works with an animated array
```

**Generated Features:**
- 📊 Visual array with animated elements
- 📈 Progress bar showing completion
- 🎮 Interactive controls (Start Sort, Reset, Next Step)
- 🔵 Blue highlighting of comparing elements
- ✅ Success message when sorted

**Code Highlights:**
```tsx
interface Step {
  index1: number;
  index2: number;
  array: number[];
}

export default function ShowingBubbleSortArrayViz() {
  const [steps, setSteps] = useState<Step[]>([]);

  // Step-by-step visualization with proper types
  // Framer Motion for smooth animations
  // Progress tracking
}
```

---

## Technical Implementation

### AI System Prompt Enhancements

The backend AI prompt includes:

**Framer Motion Rules:**
```javascript
- Use EITHER animate/initial OR variants, NEVER both together
- Simple animation: <motion.div animate={{ scale: 1 }} initial={{ scale: 0 }}>
- Keep it simple - don't over-complicate animations
- Transition goes inside animate: animate={{ scale: 1, transition: { duration: 0.5 } }}
```

**Lucide Icon Rules:**
```javascript
- Icon names do NOT have "Icon" prefix
- Icon names are PascalCase: "ArrowUpDown" NOT "Sort"
- Import like: import { ArrowUpDown, Play } from 'lucide-react'
```

---

## Security Considerations

- Generated code runs in a React Error Boundary
- Basic validation checks for required exports
- No eval() or Function() constructors used
- Code is saved to a sandboxed folder (`/src/customized/`)
- Files can be manually reviewed before execution

---

## Performance Tips

- Keep visualizations lightweight (< 500 lines)
- Use React.memo for expensive renders
- Leverage Framer Motion's `layoutId` for smooth transitions
- Use Tailwind's JIT for optimal CSS
- Avoid unnecessary re-renders with proper dependency arrays

---

## Files to Check

**Backend:**
- `/server.js` - AI system prompt configuration
- `/src/customized/` - Your generated visualizations

**Frontend:**
- `/src/App.tsx` - Main navigation and visualization loading
- `/src/utils/customVizLoader.ts` - Dynamic import logic
- `/src/components/CustomDialog.tsx` - Generation UI

---

## Contributing Custom Templates

To create reusable templates:

1. Create a well-documented visualization
2. Save it as a template in `/src/customized/`
3. Users can reference it when creating similar visualizations

---

## Future Enhancements

Potential improvements for this feature:

- [ ] Support for multiple AI models (Claude, GPT-4, etc.)
- [ ] Visualization gallery/marketplace
- [ ] Export as standalone components
- [ ] Version control for custom visualizations
- [ ] Collaborative editing
- [ ] Code playground with live preview
- [ ] Template library
- [ ] Auto-fixing common errors

---

## Support

For issues or questions:
1. Check browser console for errors (F12)
2. Review server logs in the terminal
3. Verify Ollama is responding: `curl http://localhost:11434/api/tags`
4. Check the generated code in `/src/customized/`
5. Run build to check TypeScript errors: `npm run build`

---

**Happy Visualizing! 🚀**
