import * as tf from '@tensorflow/tfjs';

// Load the models
const loadModels = async () => {
  const models = [];
  try {
    const resUNet = await tf.loadLayersModel('/models/resUNet/model.json');
    const attentionUNet = await tf.loadLayersModel('/models/attentionUNet/model.json');
    const unetPlusPlus = await tf.loadLayersModel('/models/unetPlusPlus/model.json');
    
    models.push(resUNet, attentionUNet, unetPlusPlus);
  } catch (error) {
    console.error('Failed to load models:', error);
  }
  return models;
};

// Preprocess image
const preprocessImage = async (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 128, 128);
        
        const imageData = ctx.getImageData(0, 0, 128, 128);
        const tensor = tf.browser.fromPixels(imageData, 1)
          .toFloat()
          .div(255.0)
          .expandDims(0);
          
        resolve(tensor);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
};

// Main analysis function
export const analyzeImage = async (file, params) => {
  const models = await loadModels();
  if (models.length === 0) {
    throw new Error('No models available');
  }

  const tensor = await preprocessImage(file);
  const predictions = await Promise.all(
    models.map(model => model.predict(tensor))
  );

  // Process predictions
  const results = predictions.map((pred, idx) => {
    const mask = pred.dataSync();
    const percentage = calculateAffectedArea(mask);
    return {
      modelName: ['ResU-Net', 'Attention U-Net', 'U-Net++'][idx],
      percentage,
      cancerous: percentage > params.threshold
    };
  });

  const avgPercentage = results.reduce((acc, curr) => acc + curr.percentage, 0) / results.length;
  const votes = results.filter(r => r.cancerous).length;
  const cancerous = votes > models.length / 2;

  return {
    cancerous,
    percentage: avgPercentage,
    confidence: calculateConfidence(results),
    votes,
    totalModels: models.length,
    individualResults: results,
    stage: cancerous ? determineStage(avgPercentage) : null
  };
};

// Helper functions
const calculateAffectedArea = (mask) => {
  const threshold = 0.5;
  const affected = mask.filter(p => p > threshold).length;
  return (affected / mask.length) * 100;
};

const calculateConfidence = (results) => {
  return results.reduce((acc, curr) => acc + curr.percentage, 0) / results.length;
};

const determineStage = (percentage) => {
  if (percentage < 5) return { stage: 1, name: 'Stage I', risk: 'LOW' };
  if (percentage < 15) return { stage: 2, name: 'Stage II', risk: 'MODERATE' };
  if (percentage < 30) return { stage: 3, name: 'Stage III', risk: 'HIGH' };
  return { stage: 4, name: 'Stage IV', risk: 'CRITICAL' };
};