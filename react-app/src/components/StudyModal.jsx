import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

export default function StudyModal({ isOpen, topic, content, onClose }) {
    const overlayRef = useRef(null);

    // Scroll to top when modal opens
    useEffect(() => {
        if (isOpen && overlayRef.current) {
            // Use setTimeout to ensure DOM is ready
            setTimeout(() => {
                if (overlayRef.current) {
                    overlayRef.current.scrollTop = 0;
                }
            }, 0);
        }
    }, [isOpen, topic]);

    if (!isOpen || !content) return null;

    return (
        <div
            className="modal-overlay"
            onClick={onClose}
            style={{
                position: 'fixed',
                inset: 0,
                background: 'rgba(0, 0, 0, 0.95)',
                backdropFilter: 'blur(10px)',
                zIndex: 100,
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'center',
                overflowY: 'auto',
                padding: '40px 20px'
            }}
        >
            <div
                key={topic}
                onClick={e => e.stopPropagation()}
                style={{
                    maxWidth: '900px',
                    width: '100%',
                    background: 'var(--system-panel)',
                    border: '1px solid var(--neon-purple)',
                    borderRadius: '8px',
                    boxShadow: '0 0 60px rgba(188, 19, 254, 0.2)',
                    position: 'relative'
                }}
            >
                {/* Header */}
                <div style={{
                    padding: '24px 32px',
                    borderBottom: '1px solid var(--border-color)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    position: 'sticky',
                    top: 0,
                    background: 'var(--system-panel)',
                    zIndex: 10
                }}>
                    <div>
                        <div style={{ fontSize: '10px', color: 'var(--neon-purple)', letterSpacing: '0.3em', marginBottom: '4px' }}>
                            GRIMOIRE ENTRY
                        </div>
                        <h1 style={{ fontSize: '24px', fontWeight: 'bold' }}>{topic}</h1>
                    </div>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'transparent',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-secondary)',
                            padding: '8px 16px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '12px'
                        }}
                    >
                        ✕ CLOSE
                    </button>
                </div>

                {/* Content */}
                <div
                    className="study-content"
                    style={{
                        padding: '32px',
                        fontSize: '15px',
                        lineHeight: '1.8'
                    }}
                >
                    <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                            h1: ({ children }) => (
                                <h1 style={{
                                    fontSize: '28px',
                                    fontWeight: 'bold',
                                    marginTop: '32px',
                                    marginBottom: '16px',
                                    color: 'var(--neon-blue)',
                                    borderBottom: '2px solid var(--border-color)',
                                    paddingBottom: '8px'
                                }}>{children}</h1>
                            ),
                            h2: ({ children }) => (
                                <h2 style={{
                                    fontSize: '22px',
                                    fontWeight: 'bold',
                                    marginTop: '28px',
                                    marginBottom: '12px',
                                    color: 'var(--neon-purple)'
                                }}>{children}</h2>
                            ),
                            h3: ({ children }) => (
                                <h3 style={{
                                    fontSize: '18px',
                                    fontWeight: 'bold',
                                    marginTop: '20px',
                                    marginBottom: '8px',
                                    color: 'var(--text-primary)'
                                }}>{children}</h3>
                            ),
                            p: ({ children }) => (
                                <p style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>{children}</p>
                            ),
                            ul: ({ children }) => (
                                <ul style={{ marginBottom: '16px', paddingLeft: '24px' }}>{children}</ul>
                            ),
                            ol: ({ children }) => (
                                <ol style={{ marginBottom: '16px', paddingLeft: '24px' }}>{children}</ol>
                            ),
                            li: ({ children }) => (
                                <li style={{ marginBottom: '8px', color: 'var(--text-primary)' }}>{children}</li>
                            ),
                            code: ({ node, inline, className, children, ...props }) => {
                                const match = /language-(\w+)/.exec(className || '');
                                if (!inline && match) {
                                    return (
                                        <pre style={{
                                            background: '#0a0a0a',
                                            padding: '16px',
                                            borderRadius: '8px',
                                            overflow: 'auto',
                                            marginBottom: '16px',
                                            border: '1px solid var(--border-color)'
                                        }}>
                                            <code style={{ fontSize: '13px', color: '#e0e0e0' }} {...props}>
                                                {children}
                                            </code>
                                        </pre>
                                    );
                                } else if (!inline) {
                                    return (
                                        <pre style={{
                                            background: '#0a0a0a',
                                            padding: '16px',
                                            borderRadius: '8px',
                                            overflow: 'auto',
                                            marginBottom: '16px',
                                            border: '1px solid var(--border-color)'
                                        }}>
                                            <code style={{ fontSize: '13px', color: '#e0e0e0' }} {...props}>
                                                {children}
                                            </code>
                                        </pre>
                                    );
                                }
                                return (
                                    <code style={{
                                        background: 'var(--card-bg)',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        fontSize: '13px',
                                        color: 'var(--neon-blue)'
                                    }} {...props}>
                                        {children}
                                    </code>
                                );
                            },
                            pre: ({ children }) => <>{children}</>,
                            blockquote: ({ children }) => (
                                <blockquote style={{
                                    borderLeft: '4px solid var(--neon-gold)',
                                    paddingLeft: '16px',
                                    margin: '16px 0',
                                    background: 'rgba(255, 215, 0, 0.05)',
                                    padding: '12px 16px',
                                    borderRadius: '0 4px 4px 0'
                                }}>{children}</blockquote>
                            ),
                            strong: ({ children }) => (
                                <strong style={{ color: 'var(--neon-blue)', fontWeight: 'bold' }}>{children}</strong>
                            ),
                            hr: () => (
                                <hr style={{
                                    border: 'none',
                                    borderTop: '1px solid var(--border-color)',
                                    margin: '32px 0'
                                }} />
                            ),
                            table: ({ children }) => (
                                <div style={{ overflowX: 'auto', marginBottom: '16px' }}>
                                    <table style={{
                                        width: '100%',
                                        borderCollapse: 'collapse',
                                        border: '1px solid var(--border-color)'
                                    }}>{children}</table>
                                </div>
                            ),
                            thead: ({ children }) => (
                                <thead style={{ background: 'var(--card-bg)' }}>{children}</thead>
                            ),
                            tbody: ({ children }) => (
                                <tbody>{children}</tbody>
                            ),
                            tr: ({ children }) => (
                                <tr style={{ borderBottom: '1px solid var(--border-color)' }}>{children}</tr>
                            ),
                            th: ({ children }) => (
                                <th style={{
                                    padding: '12px 16px',
                                    textAlign: 'left',
                                    fontWeight: 'bold',
                                    color: 'var(--neon-blue)',
                                    borderBottom: '2px solid var(--border-color)',
                                    borderRight: '1px solid var(--border-color)'
                                }}>{children}</th>
                            ),
                            td: ({ children }) => (
                                <td style={{
                                    padding: '12px 16px',
                                    borderRight: '1px solid var(--border-color)',
                                    verticalAlign: 'top'
                                }}>{children}</td>
                            )
                        }}
                    >
                        {content}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
    );
}
