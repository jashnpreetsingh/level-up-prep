// Content Index - Maps topic names to their content
import { mlContent } from './mlContent';
import { nlpContent } from './nlpContent';
import { mathContent } from './mathContent';

// Merge all content
export const allContent = {
    ...mlContent,
    ...nlpContent,
    ...mathContent
};

// Topic name normalization for fuzzy matching
const normalizeTitle = (title) => {
    return title.toLowerCase()
        .replace(/[^a-z0-9\s]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
};

// Get content for a topic (with fuzzy matching)
export function getTopicContent(topic) {
    // Direct match
    if (allContent[topic]) {
        return allContent[topic];
    }

    // Normalize and try to match
    const normalizedInput = normalizeTitle(topic);

    for (const [key, content] of Object.entries(allContent)) {
        if (normalizeTitle(key) === normalizedInput) {
            return content;
        }
        // Partial match
        if (normalizedInput.includes(normalizeTitle(key)) ||
            normalizeTitle(key).includes(normalizedInput)) {
            return content;
        }
    }

    return null;
}

// Get list of all available topics
export function getAvailableTopics() {
    return Object.keys(allContent);
}

// Check if topic has content
export function hasContent(topic) {
    return getTopicContent(topic) !== null;
}

export default allContent;
