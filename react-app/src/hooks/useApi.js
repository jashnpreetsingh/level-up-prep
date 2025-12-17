const API_BASE = 'http://localhost:3001/api';

export async function fetchState() {
    const res = await fetch(`${API_BASE}/state`);
    if (!res.ok) throw new Error('Failed to fetch state');
    return res.json();
}

export async function saveState(state) {
    const res = await fetch(`${API_BASE}/state`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state)
    });
    if (!res.ok) throw new Error('Failed to save state');
    return res.json();
}

export async function completeProblem(id, data) {
    const res = await fetch(`${API_BASE}/completed/${encodeURIComponent(id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error('Failed to complete problem');
    return res.json();
}

export async function markReviewed(id) {
    const res = await fetch(`${API_BASE}/reviewed/${encodeURIComponent(id)}`, {
        method: 'POST'
    });
    if (!res.ok) throw new Error('Failed to mark reviewed');
    return res.json();
}

export async function saveSlot(slot, data) {
    const res = await fetch(`${API_BASE}/saves/${slot}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data })
    });
    if (!res.ok) throw new Error('Failed to save slot');
    return res.json();
}

export async function loadSlot(slot) {
    const res = await fetch(`${API_BASE}/saves/${slot}`);
    if (!res.ok) throw new Error('Failed to load slot');
    return res.json();
}

export async function resetAll() {
    const res = await fetch(`${API_BASE}/reset`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to reset');
    return res.json();
}
