import React from 'react';
import Plot from 'react-plotly.js';

const PredictionChart = ({ predictions }) => {
  if (!predictions) {
    return null;
  }

  // Format the prediction data for Plotly
  const propertyNames = Object.keys(predictions);
  const propertyValues = Object.values(predictions);

  const data = [
    {
      x: propertyNames,
      y: propertyValues,
      type: 'bar',
      marker: {
        color: 'rgb(58, 128, 230)'
      }
    }
  ];

  const layout = {
    title: 'Predicted Material Properties',
    xaxis: {
      title: 'Property'
    },
    yaxis: {
      title: 'Predicted Value (Normalized)'
    },
    height: 400,
    width: 600,
    margin: {
      l: 50,
      r: 50,
      b: 100,
      t: 100,
      pad: 4
    }
  };

  return (
    <Plot
      data={data}
      layout={layout}
      style={{ width: '100%' }}
    />
  );
};

export default PredictionChart;