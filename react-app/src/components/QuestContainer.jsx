import { useState, useMemo } from 'react';
import { getQuestData, generateId } from '../data/problemData';
import { dsaSchedule, mlSchedule, mathSchedule, behavioralSchedule } from '../data/scheduleData';
import { hasContent } from '../data/content';

const CONTEXTS = ['DSA', 'ML', 'MATH', 'BEHAVIORAL'];

export default function QuestContainer({ completed, onStartHunt, currentContext, setCurrentContext, onOpenStudy }) {
    const [searchTerm, setSearchTerm] = useState('');
    const [difficultyFilter, setDifficultyFilter] = useState('');
    const [statusFilter, setStatusFilter] = useState('');

    const scheduleData = useMemo(() => {
        switch (currentContext) {
            case 'ML': return mlSchedule;
            case 'MATH': return mathSchedule;
            case 'BEHAVIORAL': return behavioralSchedule;
            default: return dsaSchedule;
        }
    }, [currentContext]);

    const getTypeLabel = (type) => {
        let label = type.replace(/_/g, ' ');
        if (type.includes('GRAPH')) return '🗺️ ' + label;
        if (type.includes('DP')) return '💎 ' + label;
        if (type.includes('TREE')) return '🌳 ' + label;
        if (type.includes('BINARY')) return '🔍 ' + label;
        if (type.includes('SLIDING')) return '🪟 ' + label;
        if (type.includes('HEAP')) return '⛰️ ' + label;
        if (type.includes('DESIGN')) return '🏗️ ' + label;
        if (type.includes('BACKTRACK')) return '♟️ ' + label;
        if (type.includes('NLP_CODE')) return '📝 ' + label;
        if (type.includes('GOOGLEYNESS')) return '💭 ' + label;
        if (type.includes('RAID')) return '⚔️ RAID: ' + label.replace('RAID ', '');
        if (type === 'MOCK') return '🎯 MOCK INTERVIEW';
        return label;
    };

    const getDiffLabel = (rank) => {
        switch (rank) {
            case 'S': return { label: 'LEGENDARY', class: 'diff-S' };
            case 'A': return { label: 'HARD', class: 'diff-A' };
            case 'B': return { label: 'MEDIUM', class: 'diff-B' };
            default: return { label: 'EASY', class: 'diff-C' };
        }
    };

    const getLeetCodeLink = (problem) => {
        if (currentContext !== 'DSA') return null;
        if (problem.startsWith('(') || problem.startsWith('Subtopics:')) return null;

        const slug = problem.toLowerCase()
            .replace(/[^a-z0-9\s]/g, '')
            .replace(/\s+/g, '-')
            .replace(/^-+|-+$/g, '');

        return slug ? `https://leetcode.com/problems/${slug}/` : null;
    };

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                <h2 style={{
                    color: 'var(--neon-blue)',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    letterSpacing: '0.2em',
                    borderLeft: '4px solid var(--neon-blue)',
                    paddingLeft: '12px'
                }}>
                    ACTIVE DUNGEONS
                </h2>

                <div style={{ display: 'flex', gap: '8px' }}>
                    {CONTEXTS.map(ctx => (
                        <button
                            key={ctx}
                            onClick={() => setCurrentContext(ctx)}
                            className={`tab ${currentContext === ctx ? 'active' : ''}`}
                        >
                            {ctx === 'BEHAVIORAL' ? 'GOOGLEYNESS' : ctx}
                        </button>
                    ))}
                </div>
            </div>

            {/* Filters */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
                <input
                    type="text"
                    placeholder="Search problems..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{ flex: 1, minWidth: '200px' }}
                />
                <select value={difficultyFilter} onChange={(e) => setDifficultyFilter(e.target.value)}>
                    <option value="">All Difficulties</option>
                    <option value="S">Legendary (S)</option>
                    <option value="A">Hard (A)</option>
                    <option value="B">Medium (B)</option>
                    <option value="C">Easy (C)</option>
                </select>
                <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                    <option value="">All Status</option>
                    <option value="completed">Completed</option>
                    <option value="pending">Pending</option>
                </select>
            </div>

            {/* Schedule */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '500px' }}>
                {scheduleData.map((day, dayIdx) => {
                    const isRaid = day.type.includes('RAID');
                    const isMock = day.type === 'MOCK';

                    return (
                        <div
                            key={dayIdx}
                            className="panel"
                            style={{
                                padding: '20px',
                                borderColor: isRaid || isMock ? 'var(--neon-red)' : undefined,
                                background: isMock ? 'rgba(255, 42, 42, 0.05)' : isRaid ? 'rgba(188, 19, 254, 0.05)' : undefined
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                <div>
                                    <span style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--text-secondary)' }}>
                                        {day.date}
                                    </span>
                                    <span style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--text-secondary)', marginLeft: '8px' }}>
                                        {getTypeLabel(day.type)}
                                    </span>
                                    {day.topic && (
                                        <span style={{ fontSize: '12px', color: 'var(--neon-purple)', marginLeft: '8px' }}>
                                            — {day.topic}
                                        </span>
                                    )}
                                </div>
                                {isRaid && (
                                    <span style={{ color: 'var(--neon-purple)', fontWeight: 900, fontSize: '12px' }} className="pulse">
                                        ⚔️ RAID DAY
                                    </span>
                                )}
                                {isMock && (
                                    <span style={{ color: 'var(--neon-red)', fontWeight: 900, fontSize: '12px' }} className="pulse">
                                        ⚠ CRITICAL
                                    </span>
                                )}
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                {day.problems.map((problem, probIdx) => {
                                    const pid = generateId(currentContext, problem);
                                    const isDone = completed[pid];
                                    const meta = getQuestData(problem, currentContext);
                                    const isSubtopic = problem.startsWith('Subtopics:');
                                    const diff = getDiffLabel(meta.rank);
                                    const leetcodeLink = getLeetCodeLink(problem);

                                    // Apply filters
                                    if (searchTerm && !problem.toLowerCase().includes(searchTerm.toLowerCase())) return null;
                                    if (difficultyFilter && meta.rank !== difficultyFilter) return null;
                                    if (statusFilter === 'completed' && !isDone) return null;
                                    if (statusFilter === 'pending' && isDone) return null;

                                    return (
                                        <div
                                            key={probIdx}
                                            className="problem-row"
                                            style={{ marginLeft: isSubtopic ? '16px' : 0, opacity: isSubtopic ? 0.75 : 1 }}
                                        >
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                                                <div className={`problem-dot ${isDone ? 'completed' : ''}`} />
                                                <span style={{
                                                    color: isDone ? 'var(--text-secondary)' : 'var(--text-primary)',
                                                    textDecoration: isDone ? 'line-through' : 'none',
                                                    fontStyle: isSubtopic ? 'italic' : 'normal',
                                                    fontSize: '13px'
                                                }}>
                                                    {problem}
                                                </span>

                                                {!isSubtopic && currentContext !== 'BEHAVIORAL' && (
                                                    <span className={diff.class} style={{ fontSize: '10px', fontWeight: 'bold' }}>
                                                        [{diff.label}]
                                                    </span>
                                                )}

                                                {leetcodeLink && (
                                                    <a
                                                        href={leetcodeLink}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        style={{ color: 'var(--neon-blue)', fontSize: '10px' }}
                                                        onClick={(e) => e.stopPropagation()}
                                                    >
                                                        🔗
                                                    </a>
                                                )}
                                            </div>

                                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                                {isDone ? (
                                                    <span style={{ color: 'var(--neon-blue)', fontSize: '12px', fontWeight: 'bold' }}>
                                                        CLEARED {isDone.retreat ? '(RETREAT)' : ''}
                                                    </span>
                                                ) : isSubtopic ? (
                                                    <span style={{ color: 'var(--text-secondary)', fontSize: '9px' }}>STUDY</span>
                                                ) : (
                                                    <>
                                                        {/* Study button for ML/MATH topics with content */}
                                                        {(currentContext === 'ML' || currentContext === 'MATH') &&
                                                            hasContent(problem) && (
                                                                <button
                                                                    className="btn"
                                                                    onClick={() => onOpenStudy(problem)}
                                                                    style={{
                                                                        fontSize: '10px',
                                                                        padding: '4px 12px',
                                                                        borderColor: 'var(--neon-purple)',
                                                                        color: 'var(--neon-purple)'
                                                                    }}
                                                                >
                                                                    📖 STUDY
                                                                </button>
                                                            )}
                                                        <button
                                                            className="btn"
                                                            onClick={() => onStartHunt(problem, currentContext, meta)}
                                                            style={{ fontSize: '10px', padding: '4px 12px' }}
                                                        >
                                                            {currentContext === 'BEHAVIORAL' ? '💭 REFLECT' : '▶ START HUNT'}
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
