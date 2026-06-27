import * as fs from 'fs';
import { fileURLToPath } from 'url';
import * as path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const loadDataset = () => {
  const files = [
    { file: '../datasets/features.json', cat: 'FEATURES' },
    { file: '../datasets/bug_fixed.json', cat: 'BUG_FIXES' },
    { file: '../datasets/maintenance.json', cat: 'MAINTENANCE' },
    { file: '../datasets/junk.json', cat: 'UNCATEGORIZED' }
  ];

  let dataset = [];

  files.forEach(({ file, cat }) => {
    const filePath = path.resolve(__dirname, file);
    
    try {
      if (fs.existsSync(filePath)) {
        const raw = fs.readFileSync(filePath, 'utf8');
        const content = JSON.parse(raw);

        const formatted = content.map((item, index) => {
          if (!item) return null;
          
          let result;
          if (typeof item === 'string') {
            result = { text: item, category: cat };
          } else {
            result = { text: item.text, category: item.category || cat };
          }
          
          if (!result.text) console.warn(`⚠️ Warning: Item ke-${index} di ${file} tidak punya text!`);
          return result;
        }).filter(item => item !== null);

        dataset = [...dataset, ...formatted];
      } else {
        console.warn(`⚠️ File tidak ditemukan: ${filePath}`);
      }
    } catch (e) {
      console.error(`❌ Gagal membaca ${file}:`, e);
    }
  });

  return dataset;
};