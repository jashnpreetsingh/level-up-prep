// Problem difficulty and metadata mapping
export const PROBLEM_METADATA = {
    // Two Pointers / Arrays (Easy -> Hard progression)
    "Two Sum": { diff: 'E', time: 15, topic: 'arrays' },
    "Valid Palindrome": { diff: 'E', time: 10, topic: 'arrays' },
    "Best Time to Buy and Sell Stock": { diff: 'E', time: 15, topic: 'arrays' },
    "Trapping Rain Water": { diff: 'H', time: 30, topic: 'arrays', shadow: 'Abyssal Sentinel', rank: 'S' },

    // Sliding Window (Progressive)
    "Maximum Subarray": { diff: 'E', time: 15, topic: 'sliding' },
    "Max Consecutive Ones III": { diff: 'M', time: 20, topic: 'sliding' },
    "Minimum Window Substring": { diff: 'H', time: 30, topic: 'sliding' },
    "Sliding Window Median": { diff: 'H', time: 35, topic: 'sliding' },
    "Longest Continuous Subarray Limit": { diff: 'H', time: 30, topic: 'sliding' },
    "Shortest Subarray Sum >= K": { diff: 'H', time: 35, topic: 'sliding' },
    "Constrained Subsequence Sum": { diff: 'H', time: 35, topic: 'sliding' },
    "Max Value of Equation": { diff: 'H', time: 30, topic: 'sliding' },

    // Binary Search (Progressive)
    "First Missing Positive": { diff: 'H', time: 25, topic: 'binary' },
    "Koko Eating Bananas": { diff: 'M', time: 20, topic: 'binary' },
    "Capacity To Ship Packages": { diff: 'M', time: 25, topic: 'binary' },
    "Split Array Largest Sum": { diff: 'H', time: 30, topic: 'binary' },
    "Kth Smallest Element in Matrix": { diff: 'H', time: 30, topic: 'binary' },
    "Median of Two Sorted Arrays": { diff: 'H', time: 40, topic: 'binary', shadow: 'Igris', rank: 'S' },
    "Smallest Good Base": { diff: 'H', time: 35, topic: 'binary' },
    "Min Number of Refueling Stops": { diff: 'H', time: 30, topic: 'binary' },

    // Linked List (Progressive)
    "Reverse Linked List": { diff: 'E', time: 10, topic: 'arrays' },
    "Linked List Random Node": { diff: 'M', time: 20, topic: 'design' },
    "Merge K Sorted Lists": { diff: 'H', time: 25, topic: 'heap', shadow: 'Lich', rank: 'A' },

    // Tree/Graph BFS (Progressive)
    "Invert Binary Tree": { diff: 'E', time: 10, topic: 'tree' },
    "Number of Islands": { diff: 'M', time: 20, topic: 'graph' },
    "Course Schedule II": { diff: 'M', time: 25, topic: 'graph' },
    "Serialize Binary Tree": { diff: 'H', time: 30, topic: 'tree' },
    "Word Ladder": { diff: 'H', time: 30, topic: 'graph' },
    "Word Search": { diff: 'M', time: 25, topic: 'backtrack' },
    "Word Search II": { diff: 'H', time: 35, topic: 'trie' },
    "Word Break II": { diff: 'H', time: 30, topic: 'dp' },
    "Alien Dictionary": { diff: 'H', time: 30, topic: 'graph', shadow: 'Beru', rank: 'S' },
    "All Nodes Distance K": { diff: 'M', time: 25, topic: 'tree' },
    "Bus Routes": { diff: 'H', time: 35, topic: 'graph' },
    "Critical Connections": { diff: 'H', time: 35, topic: 'graph' },

    // Dynamic Programming (Progressive)
    "Climbing Stairs": { diff: 'E', time: 10, topic: 'dp' },
    "House Robber": { diff: 'E', time: 15, topic: 'dp' },
    "Edit Distance": { diff: 'M', time: 25, topic: 'dp' },
    "Longest Increasing Subsequence": { diff: 'M', time: 25, topic: 'dp' },
    "Regular Expression Matching": { diff: 'H', time: 35, topic: 'dp' },
    "Wildcard Matching": { diff: 'H', time: 30, topic: 'dp' },

    // System Design Problems
    "Design Hit Counter": { diff: 'M', time: 30, topic: 'design' },
    "Design Twitter": { diff: 'M', time: 35, topic: 'design', shadow: 'Iron Fortress', rank: 'S' },
    "Design Search Autocomplete": { diff: 'H', time: 40, topic: 'design' },
    "LFU Cache": { diff: 'H', time: 35, topic: 'design' },

    // Advanced Mixed
    "Robot Room Cleaner": { diff: 'H', time: 45, topic: 'design', shadow: 'Automaton', rank: 'S' },
    "Web Crawler Multithreaded": { diff: 'H', time: 35, topic: 'design', shadow: 'Spider Queen', rank: 'S' },
    "The Skyline Problem": { diff: 'H', time: 35, topic: 'heap', shadow: 'Kaisel', rank: 'S' },
};

export const LEGENDARY_MAPPING = {
    "Trapping Rain Water": { rank: 'S', limit: 25, type: 'Commander', shadow: 'Abyssal Sentinel' },
    "Median of Two Sorted Arrays": { rank: 'S', limit: 40, type: 'Commander', shadow: 'Igris' },
    "The Skyline Problem": { rank: 'S', limit: 35, type: 'Commander', shadow: 'Kaisel' },
    "Alien Dictionary": { rank: 'S', limit: 30, type: 'Commander', shadow: 'Beru' },
    "Word Ladder II": { rank: 'A', limit: 30, type: 'Assassin', shadow: 'Phantom' },
    "Design Twitter": { rank: 'S', limit: 40, type: 'Tank', shadow: 'Iron Fortress' },
    "Merge K Sorted Lists": { rank: 'A', limit: 25, type: 'Mage', shadow: 'Lich' },
    "Robot Room Cleaner": { rank: 'S', limit: 45, type: 'Assassin', shadow: 'Automaton' },
    "Web Crawler Multithreaded": { rank: 'S', limit: 35, type: 'Tank', shadow: 'Spider Queen' },
};

export function getQuestData(title, context) {
    if (context === 'BEHAVIORAL') return { rank: 'B', limit: 15, type: 'Rune', time: 'O(1)' };
    if (context !== 'DSA') return { rank: 'B', limit: 60, type: context === 'ML' ? 'Construct' : 'Rune', time: 'O(1)' };

    const meta = PROBLEM_METADATA[title];
    if (!meta) return { time: 'O(N)', space: 'O(N)', limit: 20, rank: 'B', type: 'Infantry' };

    let data = {
        time: 'O(N)',
        space: 'O(1)',
        limit: meta.time || 20,
        rank: meta.diff === 'H' ? 'A' : (meta.diff === 'M' ? 'B' : 'C'),
        type: 'Infantry',
        topic: meta.topic
    };

    if (LEGENDARY_MAPPING[title]) {
        const legendary = LEGENDARY_MAPPING[title];
        data.rank = legendary.rank;
        data.type = legendary.type;
        data.shadow = legendary.shadow;
    } else {
        if (meta.topic === 'graph' || meta.topic === 'tree') data.type = 'Assassin';
        else if (meta.topic === 'dp') data.type = 'Mage';
        else if (meta.topic === 'design') data.type = 'Tank';
        else if (meta.topic === 'heap' || meta.topic === 'binary') data.type = 'Mage';
    }

    if (meta.shadow) {
        data.shadow = meta.shadow;
        data.rank = meta.rank;
    }

    return data;
}

export function generateId(ctx, val) {
    return ctx + '_' + String(val).replace(/[^a-zA-Z0-9]/g, '_');
}
