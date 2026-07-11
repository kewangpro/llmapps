import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from './ui/dialog';
import { Loader2, Sparkles, Save, X } from 'lucide-react';

interface CustomDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onVisualizationCreated: () => void;
}

export default function CustomDialog({ open, onOpenChange, onVisualizationCreated }: CustomDialogProps) {
  const [prompt, setPrompt] = useState('');
  const [generatedCode, setGeneratedCode] = useState('');
  const [componentName, setComponentName] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setIsGenerating(true);
    setError('');
    setGeneratedCode('');
    setValidationErrors([]);

    try {
      const response = await fetch('http://localhost:3005/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate visualization');
      }

      const data = await response.json();

      setGeneratedCode(data.code);
      setComponentName(data.componentName);

      if (!data.validation.isValid) {
        setValidationErrors(data.validation.errors);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate visualization');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSave = async () => {
    if (!generatedCode || !componentName) {
      setError('No code to save');
      return;
    }

    setIsSaving(true);
    setError('');

    try {
      const response = await fetch('http://localhost:3005/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: generatedCode,
          fileName: componentName,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to save visualization');
      }

      // Success! Reset state
      setPrompt('');
      setGeneratedCode('');
      setComponentName('');
      setValidationErrors([]);
      onOpenChange(false);

      // Notify parent to reload visualizations list
      onVisualizationCreated();

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save visualization');
    } finally {
      setIsSaving(false);
    }
  };

  const handleClose = () => {
    if (!isGenerating && !isSaving) {
      setPrompt('');
      setGeneratedCode('');
      setComponentName('');
      setError('');
      setValidationErrors([]);
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="text-yellow-500" size={24} />
            Create Custom Visualization
          </DialogTitle>
          <DialogDescription>
            Describe the visualization you want to create, and AI will generate it for you.
          </DialogDescription>
        </DialogHeader>

        <div className="px-6 py-4 space-y-4">
          {/* Prompt Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Visualization Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., 'Create a visualization showing how binary search works with an interactive array'"
              className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              disabled={isGenerating || isSaving}
            />
          </div>

          {/* Generate Button */}
          {!generatedCode && (
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim()}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  Generate Visualization
                </>
              )}
            </button>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {error}
            </div>
          )}

          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm font-medium text-yellow-800 mb-1">Validation Warnings:</p>
              <ul className="text-sm text-yellow-700 list-disc list-inside">
                {validationErrors.map((err, idx) => (
                  <li key={idx}>{err}</li>
                ))}
              </ul>
              <p className="text-xs text-yellow-600 mt-2">
                The code may still work, but consider regenerating if issues occur.
              </p>
            </div>
          )}

          {/* Code Preview */}
          {generatedCode && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-700">
                  Generated Code: <span className="text-blue-600">{componentName}.tsx</span>
                </label>
                <button
                  onClick={() => {
                    setGeneratedCode('');
                    setComponentName('');
                    setValidationErrors([]);
                  }}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Clear
                </button>
              </div>
              <div className="relative">
                <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto max-h-96 text-xs font-mono">
                  <code>{generatedCode}</code>
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Footer with Save Button */}
        {generatedCode && (
          <DialogFooter>
            <button
              onClick={handleClose}
              disabled={isSaving}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
            >
              <X size={18} />
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
            >
              {isSaving ? (
                <>
                  <Loader2 className="animate-spin" size={18} />
                  Saving...
                </>
              ) : (
                <>
                  <Save size={18} />
                  Save Visualization
                </>
              )}
            </button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
