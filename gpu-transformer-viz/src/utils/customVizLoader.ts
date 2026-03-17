// Dynamic loader for custom visualizations
import { lazy } from 'react';

export interface CustomVizMetadata {
  name: string;
  icon: string;
}

export interface CustomVizModule {
  default: React.ComponentType;
  metadata: CustomVizMetadata;
}

export async function loadCustomVisualizations() {
  const visualizations: Array<{
    id: string;
    name: string;
    icon: string;
    component: React.LazyExoticComponent<React.ComponentType>;
  }> = [];

  try {
    // Fetch list of files from backend (includes newly created files)
    console.log('🔍 Fetching file list from backend...');
    const response = await fetch('http://localhost:3005/api/list');
    if (!response.ok) {
      console.warn('Failed to fetch visualization list from backend');
      return visualizations;
    }

    const data = await response.json();
    const files: string[] = data.files || [];
    console.log('📁 Files found:', files);

    // Dynamically import each file
    for (const file of files) {
      try {
        const fileName = file.replace('.tsx', '');

        // Skip .gitkeep and other non-component files
        if (fileName.startsWith('.')) continue;

        console.log(`📦 Loading ${file}...`);

        // Create lazy component with dynamic import
        // Add timestamp to bust cache for newly created files
        const timestamp = Date.now();
        const LazyComponent = lazy(() =>
          import(/* @vite-ignore */ `/src/customized/${file}?t=${timestamp}`)
            .catch(err => {
              console.error(`❌ Failed to load ${file}:`, err);
              // Return fallback component
              return {
                default: () => null,
                metadata: { name: 'Error', icon: 'AlertTriangle' }
              };
            })
        );

        // Load metadata immediately for navigation
        const module = await import(/* @vite-ignore */ `/src/customized/${file}?t=${timestamp}`);

        if (module.metadata) {
          console.log(`✅ Loaded: ${module.metadata.name} (icon: ${module.metadata.icon})`);
          visualizations.push({
            id: fileName,
            name: module.metadata.name,
            icon: module.metadata.icon,
            component: LazyComponent,
          });
        } else {
          console.warn(`⚠️ No metadata found in ${file}`);
        }
      } catch (error) {
        console.error(`❌ Failed to load custom visualization ${file}:`, error);
      }
    }

    console.log(`📊 Total visualizations loaded: ${visualizations.length}`);
  } catch (error) {
    console.error('Failed to load custom visualizations:', error);
  }

  return visualizations;
}

// No longer needed - dynamic imports pick up new files automatically
export function reloadForNewVisualization() {
  // Force reload to ensure clean state
  window.location.reload();
}
