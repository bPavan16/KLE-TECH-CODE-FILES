import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const app = express();
app.use(express.json());

// Serve static files
app.use(express.static('public'));

app.post('/api/write', (req, res) => {
    const { content, path } = req.body;

    fs.writeFile(path, content, (err) => {
        if (err) {
            return res.status(500).send('Error writing file: ' + err.message);
        }
        res.status(200).send('File written successfully.');
    });
});

app.post('/api/swap', (req, res) => {
    const { path1, path2 } = req.body;
    if (!path1 || !path2) {
        return res.status(400).send('Both file paths are required.');
    }

    try {
        const data1 = fs.readFileSync(path1, 'utf8');
        const data2 = fs.readFileSync(path2, 'utf8');

        fs.writeFileSync(path2, data1, 'utf8');
        fs.writeFileSync(path1, data2, 'utf8');

        res.status(200).send('Files have been swapped successfully.');
    } catch (error) {
        res.status(500).send('Error swapping files: ' + error.message);
    }
});

// Serve HTML pages
app.get('/swap', (req, res) => {
    res.sendFile(path.join(__dirname, 'pages', 'swap.html'));
});

app.get('/write', (req, res) => {
    res.sendFile(path.join(__dirname, 'pages', 'write.html'));
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'pages', 'home.html'));
});

const PORT = 8800;

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log(`Visit http://localhost:${PORT} to swap files`);
});
