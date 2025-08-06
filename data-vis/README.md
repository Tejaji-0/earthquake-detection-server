# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

# Earthquake Data Visualization

A React TypeScript application for visualizing global earthquake data from 1995-2023. This interactive dashboard displays earthquake information including magnitude, location, time, and various seismic parameters.

## Features

- **Interactive Data Table**: Sort and paginate through 1000+ earthquake records
- **Advanced Filtering**: Filter by magnitude range, date range, location, and alert level
- **Data Visualization**: Charts showing magnitude distribution, alert levels, and tsunami potential
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Statistics**: Summary statistics including total earthquakes, average magnitude, and tsunami events

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Chart.js** with react-chartjs-2 for data visualization
- **Papa Parse** for CSV data processing
- **Lucide React** for icons
- **CSS3** with modern responsive design

## Dataset

The application uses earthquake data spanning from 1995 to 2023, including:
- 1000+ earthquake records with magnitude 6.5+
- Geographic coordinates (latitude/longitude)
- Seismic parameters (depth, significance, alert levels)
- Tsunami potential indicators
- Location and country information

## Getting Started

### Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser to the local development URL (typically `http://localhost:5173`)

### Building for Production

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## Project Structure

```
src/
├── components/          # React components
│   ├── EarthquakeTable.tsx    # Data table component
│   ├── EarthquakeCharts.tsx   # Visualization charts
│   └── FilterPanel.tsx        # Filter controls
├── types/              # TypeScript type definitions
│   └── earthquake.ts   # Data interfaces
├── utils/              # Utility functions
│   └── dataLoader.ts   # CSV parsing and filtering
├── App.tsx             # Main application component
└── index.tsx           # Application entry point
```

## Usage

1. **Overview Tab**: View data visualizations including:
   - Magnitude distribution bar chart
   - Alert level distribution pie chart
   - Tsunami potential statistics
   - Summary statistics

2. **Data Table Tab**: Browse the complete dataset with:
   - Sortable columns
   - Pagination
   - Color-coded magnitude indicators
   - Alert level badges

3. **Filtering**: Use the filter panel to:
   - Set magnitude range (6.5 - 9.5)
   - Select date range
   - Search by location or country
   - Filter by alert level (green/yellow/red)

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style

This project uses:
- TypeScript for type safety
- ESLint for code linting
- Functional React components with hooks
- CSS modules for styling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
