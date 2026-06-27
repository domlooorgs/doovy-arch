import { CommitClassifier } from '../dist/index.min.mjs';
import { loadDataset } from "./utils/loadDataset.js";
import { shuffleArray } from "./utils/shuffleArray.js";

const classifier = new CommitClassifier({
  hiddenSize: 64,
  learningRate: 0.02,
  momentum: 0.9,
  maxGrad: 0.5,
  confidenceThreshold: 0.75,
  activation: 'swish'
});

process.on('SIGINT', () => {
  console.log('\n\n⚠️ CTRL+C terdeteksi! Sedang menyimpan progress...');
  classifier.saveModel('./training/models/doovy-source-001.json');
  console.log('✅ Progress tersimpan di checkpoint-model.json. Exiting...');
  process.exit();
});

async function main() {
  const data = shuffleArray(loadDataset());
  
  console.log("🚀 Starting Training...");
  await classifier.continueTraining({
    data: data,
    filePath: './training/models/doovy-source-001.json',
    epochs: 300,
    decayFactor: 0.1
  });
}

main();