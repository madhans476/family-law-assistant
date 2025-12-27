"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Plus, Scale, BookOpen, Loader2, MessageSquare, ExternalLink, FileText, AlertCircle, CheckCircle } from 'lucide-react';

// Types
interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
}

interface Source {
  title: string;
  url?: string;
  category?: string;
  metadata?: {
    section?: string;
    source?: string;
  };
}

interface Conversation {
  conversation_id: string;
  last_modified: string;
}

const LegalAssistChat: React.FC = () => {
  const [threads, setThreads] = useState<Conversation[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const API_BASE = 'http://localhost:8000';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  useEffect(() => {
    loadThreads();
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const loadThreads = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`);
      if (!res.ok) throw new Error('Failed to load conversations');
      const data = await res.json();
      setThreads(data.conversations || []);
    } catch (err) {
      console.error('Failed to load threads:', err);
      setError('Unable to load conversation history');
    }
  };

  const createNewThread = () => {
    const newThreadId = `conv_${Date.now()}`;
    setCurrentThreadId(newThreadId);
    setMessages([]);  // Clear messages
    setSources([]);   // Clear sources
    setStreamingMessage('');  // Clear streaming message
    setError(null);
    setInput('');  // Clear input
  };

  const loadThread = async (threadId: string) => {
    try {
      setError(null);
      setStreamingMessage('');  // Clear any streaming message
      const res = await fetch(`${API_BASE}/history/${threadId}`);
      if (!res.ok) throw new Error('Failed to load conversation');
      const data = await res.json();
      setCurrentThreadId(threadId);
      
      const formattedMessages: Message[] = data.map((msg: any) => ({
        role: msg.role === 'HumanMessage' ? 'user' : 'assistant',
        content: msg.content,
        timestamp: new Date()
      }));
      setMessages(formattedMessages);
      setSources([]);  // Clear sources when loading new thread
    } catch (err) {
      console.error('Failed to load thread:', err);
      setError('Unable to load this conversation');
    }
  };

  const deleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this conversation?')) return;
    
    try {
      const res = await fetch(`${API_BASE}/history/${threadId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete');
      loadThreads();
      if (currentThreadId === threadId) {
        setCurrentThreadId(null);
        setMessages([]);
        setSources([]);
        setStreamingMessage('');
      }
    } catch (err) {
      console.error('Failed to delete thread:', err);
      setError('Unable to delete conversation');
    }
  };

  const sendMessage = async (text: string = input) => {
    if (!text.trim() || isLoading) return;

    const threadId = currentThreadId || `conv_${Date.now()}`;
    if (!currentThreadId) {
      setCurrentThreadId(threadId);
    }

    const userMessage: Message = { 
      role: 'user', 
      content: text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingMessage('');
    setSources([]);  // Clear previous sources
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: text, 
          conversation_id: threadId 
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');
      
      const decoder = new TextDecoder();
      let accumulatedText = '';
      let sourcesReceived: Source[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'token') {
                accumulatedText += data.content;
                setStreamingMessage(accumulatedText);
              } else if (data.type === 'done') {
                if (accumulatedText) {  // Only add if there's content
                  setMessages(prev => [...prev, { 
                    role: 'assistant', 
                    content: accumulatedText,
                    timestamp: new Date()
                  }]);
                }
                setStreamingMessage('');
              } else if (data.type === 'sources') {
                sourcesReceived = data.sources || [];
                setSources(sourcesReceived);
              } else if (data.type === 'metadata') {
                setCurrentThreadId(data.conversation_id);
              } else if (data.type === 'error') {
                throw new Error(data.message);
              }
            } catch (parseErr) {
              // Ignore JSON parse errors for incomplete chunks
              console.warn('Parse error:', parseErr);
            }
          }
        }
      }

      // Final check: if we have accumulated text but no 'done' event
      if (accumulatedText && streamingMessage) {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: accumulatedText,
          timestamp: new Date()
        }]);
        setStreamingMessage('');
      }

      loadThreads();
    } catch (err) {
      console.error('Failed to send message:', err);
      setError('Failed to get response. Please try again.');
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '❌ I apologize, but I encountered an error. Please try again or rephrase your question.',
        timestamp: new Date()
      }]);
      setStreamingMessage('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (text: string): string => {
    if (!text) return '';
    
    // Bold text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Code blocks
    text = text.replace(/```(.*?)```/gs, '<pre class="bg-gray-800 p-3 rounded-lg my-2 overflow-x-auto"><code>$1</code></pre>');
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code class="bg-gray-800 px-2 py-1 rounded text-sm">$1</code>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br/>');
    
    return text;
  };

  const formatTime = (date?: Date): string => {
    if (!date) return '';
    return new Intl.DateTimeFormat('en-IN', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 overflow-hidden">
      {/* Sidebar */}
      <div className="w-80 bg-slate-950/50 backdrop-blur-xl border-r border-blue-500/20 flex flex-col shadow-2xl">
        {/* Sidebar Header */}
        <div className="p-6 border-b border-blue-500/20 bg-gradient-to-r from-blue-900/30 to-slate-900/30">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Scale className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Family Law</h1>
              <p className="text-xs text-blue-300">Legal Assistant</p>
            </div>
          </div>
          <button
            onClick={createNewThread}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-lg hover:from-blue-700 hover:to-blue-600 transition-all shadow-lg hover:shadow-blue-500/50 font-medium"
          >
            <Plus className="w-5 h-5" />
            New Consultation
          </button>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {threads.length === 0 ? (
            <div className="text-center text-gray-400 py-12 px-4">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No conversations yet</p>
              <p className="text-xs mt-1 opacity-75">Start a new consultation</p>
            </div>
          ) : (
            threads.map((thread) => (
              <div
                key={thread.conversation_id}
                onClick={() => loadThread(thread.conversation_id)}
                className={`group p-4 rounded-lg cursor-pointer transition-all ${
                  currentThreadId === thread.conversation_id
                    ? 'bg-blue-600/30 border border-blue-500 shadow-lg'
                    : 'bg-slate-800/30 hover:bg-slate-800/50 border border-transparent hover:border-blue-500/30'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <FileText className="w-4 h-4 text-blue-400 flex-shrink-0" />
                      <p className="text-white font-medium text-sm truncate">
                        {thread.conversation_id.replace('conv_', 'Case ')}
                      </p>
                    </div>
                    <p className="text-gray-400 text-xs">
                      {new Date(thread.last_modified).toLocaleDateString('en-IN', {
                        day: 'numeric',
                        month: 'short',
                        year: 'numeric'
                      })}
                    </p>
                  </div>
                  <button
                    onClick={(e) => deleteThread(thread.conversation_id, e)}
                    className="ml-2 p-2 text-gray-400 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-blue-500/20 bg-slate-900/30">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <AlertCircle className="w-4 h-4" />
            <span>Powered by AI • For information only</span>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-slate-950/50 backdrop-blur-xl border-b border-blue-500/20 p-6 shadow-xl">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl">
              <BookOpen className="w-6 h-6 text-blue-400" />
            </div>
            <div className="flex-1">
              <h2 className="text-white font-semibold text-lg">AI Legal Consultation</h2>
              <p className="text-sm text-gray-400 mt-1">
                Expert guidance on family law matters • Marriage, Divorce, Custody, Domestic Violence
              </p>
            </div>
            {currentThreadId && (
              <div className="px-4 py-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
                <p className="text-xs text-blue-400 font-medium">Active Case</p>
              </div>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {error && (
            <div className="max-w-4xl mx-auto bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-400 font-medium">Error</p>
                <p className="text-red-300 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {messages.length === 0 && !streamingMessage && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-2xl px-6">
                <div className="inline-flex p-6 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-2xl mb-6">
                  <Scale className="w-16 h-16 text-blue-400" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">Welcome to Your Legal Assistant</h3>
                <p className="text-gray-300 mb-6">
                  I'm here to help with family law matters. I'll ask relevant questions to understand your situation before providing guidance.
                </p>
                <div className="grid grid-cols-2 gap-3 text-left">
                  {[
                    'Divorce & Separation',
                    'Child Custody',
                    'Domestic Violence',
                    'Maintenance & Alimony'
                  ].map((topic) => (
                    <div key={topic} className="flex items-center gap-2 p-3 bg-slate-800/30 rounded-lg border border-blue-500/10">
                      <CheckCircle className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-gray-300">{topic}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
            >
              <div
                className={`max-w-3xl ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white'
                    : 'bg-slate-800/50 backdrop-blur-sm text-gray-100 border border-blue-500/10'
                } rounded-2xl shadow-lg overflow-hidden`}
              >
                <div className="px-6 py-4">
                  <div
                    dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                    className="prose prose-invert max-w-none prose-headings:text-white prose-p:text-gray-100 prose-strong:text-white prose-code:text-blue-300"
                  />
                </div>
                {msg.timestamp && (
                  <div className="px-6 py-2 border-t border-white/10 bg-black/10">
                    <p className="text-xs opacity-75">{formatTime(msg.timestamp)}</p>
                  </div>
                )}
              </div>
            </div>
          ))}

          {streamingMessage && (
            <div className="flex justify-start animate-fade-in">
              <div className="max-w-3xl bg-slate-800/50 backdrop-blur-sm text-gray-100 border border-blue-500/10 rounded-2xl shadow-lg overflow-hidden">
                <div className="px-6 py-4">
                  <div
                    dangerouslySetInnerHTML={{ __html: formatMessage(streamingMessage) }}
                    className="prose prose-invert max-w-none"
                  />
                  <div className="flex items-center gap-2 mt-3">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                    <span className="text-xs text-blue-400">Processing...</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {sources.length > 0 && !streamingMessage && (
            <div className="max-w-3xl mx-auto bg-gradient-to-r from-blue-500/10 to-blue-600/10 border border-blue-500/30 rounded-xl p-5 shadow-lg">
              <div className="flex items-center gap-2 mb-4">
                <BookOpen className="w-5 h-5 text-blue-400" />
                <span className="text-sm font-semibold text-blue-300">Legal References</span>
              </div>
              <div className="space-y-3">
                {sources.map((source, idx) => (
                  <div key={idx} className="bg-slate-800/30 rounded-lg p-4 hover:bg-slate-800/50 transition-colors">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        {source.url ? (
                          <a 
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-300 hover:text-blue-200 font-medium flex items-center gap-2 group"
                          >
                            <span>{source.title || `Reference ${idx + 1}`}</span>
                            <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                          </a>
                        ) : (
                          <span className="text-gray-200 font-medium">{source.title || `Reference ${idx + 1}`}</span>
                        )}
                        {source.category && (
                          <span className="inline-block mt-2 px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full">
                            {source.category}
                          </span>
                        )}
                        {source.metadata && (source.metadata.section || source.metadata.source) && (
                          <p className="text-xs text-gray-400 mt-2">
                            {source.metadata.section || source.metadata.source}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-slate-950/50 backdrop-blur-xl border-t border-blue-500/20 p-6 shadow-2xl">
          <div className="max-w-4xl mx-auto">
            <div className="relative flex items-end gap-3">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Describe your legal situation..."
                disabled={isLoading}
                rows={1}
                className="flex-1 px-5 py-4 bg-slate-800/50 backdrop-blur-sm border border-blue-500/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none disabled:opacity-50 transition-all"
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
              <button
                onClick={() => sendMessage()}
                disabled={isLoading || !input.trim()}
                className="px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-xl hover:from-blue-700 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-blue-500/50 font-medium flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span className="hidden sm:inline">Processing</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span className="hidden sm:inline">Send</span>
                  </>
                )}
              </button>
            </div>
            <div className="flex items-center justify-between mt-3 px-1">
              <p className="text-xs text-gray-400">
                Press <kbd className="px-2 py-0.5 bg-slate-700 rounded text-xs">Enter</kbd> to send, <kbd className="px-2 py-0.5 bg-slate-700 rounded text-xs">Shift+Enter</kbd> for new line
              </p>
              <p className="text-xs text-gray-500">
                AI-powered legal assistance
              </p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default LegalAssistChat;