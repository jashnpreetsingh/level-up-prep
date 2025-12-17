import express from 'express';
import cors from 'cors';
import { getState, saveState, completeProblem, markReviewed, saveSlot, loadSlot, resetAll } from './database.js';

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Get full state
app.get('/api/state', (req, res) => {
    try {
        const state = getState();
        res.json(state);
    } catch (error) {
        console.error('Error getting state:', error);
        res.status(500).json({ error: error.message });
    }
});

// Save full state
app.put('/api/state', (req, res) => {
    try {
        const result = saveState(req.body);
        res.json(result);
    } catch (error) {
        console.error('Error saving state:', error);
        res.status(500).json({ error: error.message });
    }
});

// Complete a problem
app.post('/api/completed/:id', (req, res) => {
    try {
        const result = completeProblem(req.params.id, req.body);
        res.json(result);
    } catch (error) {
        console.error('Error completing problem:', error);
        res.status(500).json({ error: error.message });
    }
});

// Mark problem as reviewed
app.post('/api/reviewed/:id', (req, res) => {
    try {
        const result = markReviewed(req.params.id);
        res.json(result);
    } catch (error) {
        console.error('Error marking reviewed:', error);
        res.status(500).json({ error: error.message });
    }
});

// Save to slot
app.post('/api/saves/:slot', (req, res) => {
    try {
        const result = saveSlot(parseInt(req.params.slot), req.body.data);
        res.json(result);
    } catch (error) {
        console.error('Error saving slot:', error);
        res.status(500).json({ error: error.message });
    }
});

// Load from slot
app.get('/api/saves/:slot', (req, res) => {
    try {
        const save = loadSlot(parseInt(req.params.slot));
        res.json(save);
    } catch (error) {
        console.error('Error loading slot:', error);
        res.status(500).json({ error: error.message });
    }
});

// Reset all data
app.post('/api/reset', (req, res) => {
    try {
        const result = resetAll();
        res.json(result);
    } catch (error) {
        console.error('Error resetting:', error);
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Shadow Monarch API running on http://localhost:${PORT}`);
});
