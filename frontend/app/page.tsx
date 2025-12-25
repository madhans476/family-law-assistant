"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Plus, Scale, BookOpen, AlertCircle, Loader2, MessageSquare, ExternalLink } from 'lucide-react';

const LegalAssistChat = () => {
  const [threads, setThreads] = useState([]);
  const [currentThreadId, setCurrentThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const messagesEndRef = useRef(null);
  const [streamingMessage, setStreamingMessage] = useState('');
  
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

  const loadThreads = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`);
      const data = await res.json();
      setThreads(data.conversations || []);
    } catch (err) {
      console.error('Failed to load threads:', err);
    }
  };

  const createNewThread = () => {
    const newThreadId = `conv_${Date.now()}`;
    setCurrentThreadId(newThreadId);
    setMessages([]);
    setSources([]);
  };

  const loadThread = async (threadId) => {
    try {
      const res = await fetch(`${API_BASE}/history/${threadId}`);
      const data = await res.json();
      setCurrentThreadId(threadId);
      
      const formattedMessages = data.map(msg => ({
        role: msg.role === 'HumanMessage' ? 'user' : 'assistant',
        content: msg.content
      }));
      setMessages(formattedMessages);
      setSources([]);
    } catch (err) {
      console.error('Failed to load thread:', err);
    }
  };

  const deleteThread = async (threadId, e) => {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE}/history/${threadId}`, { method: 'DELETE' });
      loadThreads();
      if (currentThreadId === threadId) {
        setCurrentThreadId(null);
        setMessages([]);
        setSources([]);
      }
    } catch (err) {
      console.error('Failed to delete thread:', err);
    }
  };

  const sendMessage = async (text = input) => {
    if (!text.trim() || isLoading) return;

    if (!currentThreadId) createNewThread();

    const threadId = currentThreadId || `conv_${Date.now()}`;
    if (!currentThreadId) setCurrentThreadId(threadId);

    const userMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingMessage('');
    setSources([]);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: text, 
          conversation_id: threadId 
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = '';

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
                setMessages(prev => [...prev, { 
                  role: 'assistant', 
                  content: accumulatedText 
                }]);
                setStreamingMessage('');
              } else if (data.type === 'sources') {
                setSources(data.sources || []);
              } else if (data.type === 'metadata') {
                setCurrentThreadId(data.conversation_id);
              } else if (data.type === 'error') {
                console.error('Stream error:', data.message);
                setMessages(prev => [...prev, { 
                  role: 'assistant', 
                  content: `❌ Error: ${data.message}` 
                }]);
                setStreamingMessage('');
              }
            } catch (err) {
              // Ignore JSON parse errors for incomplete chunks
            }
          }
        }
      }

      loadThreads();
    } catch (err) {
      console.error('Failed to send message:', err);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '❌ Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (text) => {
    if (!text) return '';
    
    // Bold text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Code blocks
    text = text.replace(/``````/gs, '<pre><code>$1</code></pre>');
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br/>');
    
    return text;
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Sidebar */}
      <div className="w-80 bg-black/30 backdrop-blur-md border-r border-purple-500/20 flex flex-col">
        <div className="p-4 border-b border-purple-500/20">
          <div className="flex items-center gap-2 mb-4">
            <Scale className="w-6 h-6 text-purple-400" />
            <h1 className="text-xl font-bold text-white">Family Law Assistant</h1>
          </div>
          <button
            onClick={createNewThread}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg"
          >
            <Plus className="w-5 h-5" />
            New Conversation
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {threads.length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No conversations yet</p>
            </div>
          ) : (
            threads.map((thread) => (
              <div
                key={thread.conversation_id}
                onClick={() => loadThread(thread.conversation_id)}
                className={`p-3 mb-2 rounded-lg cursor-pointer transition-all ${
                  currentThreadId === thread.conversation_id
                    ? 'bg-purple-600/30 border border-purple-500'
                    : 'bg-white/5 hover:bg-white/10 border border-transparent'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium text-sm truncate">
                      {thread.conversation_id.replace('conv_', 'Conversation ')}
                    </p>
                    <p className="text-gray-400 text-xs mt-1">
                      {new Date(thread.last_modified).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => deleteThread(thread.conversation_id, e)}
                    className="ml-2 p-1 text-gray-400 hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-black/30 backdrop-blur-md border-b border-purple-500/20 p-4">
          <div className="flex items-center gap-3">
            <BookOpen className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-white font-semibold">Family Law Legal Assistant</h2>
              <p className="text-sm text-gray-400">
                Your AI-powered family law assistant. Ask questions about marriage, divorce, custody, and more.
              </p>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && !streamingMessage && (
            <div className="text-center text-gray-400 py-12">
              <Scale className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <h3 className="text-xl font-semibold mb-2">Welcome to Family Law Assistant</h3>
              <p>Start a conversation by asking a question about family law.</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl px-6 py-4 rounded-2xl ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                    : 'bg-white/10 backdrop-blur-sm text-gray-100 border border-purple-500/20'
                }`}
              >
                <div
                  dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                  className="prose prose-invert max-w-none"
                />
              </div>
            </div>
          ))}

          {streamingMessage && (
            <div className="flex justify-start">
              <div className="max-w-3xl px-6 py-4 rounded-2xl bg-white/10 backdrop-blur-sm text-gray-100 border border-purple-500/20">
                <div
                  dangerouslySetInnerHTML={{ __html: formatMessage(streamingMessage) }}
                  className="prose prose-invert max-w-none"
                />
                <Loader2 className="w-4 h-4 animate-spin inline-block ml-2" />
              </div>
            </div>
          )}

          {sources.length > 0 && (
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <BookOpen className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-semibold text-blue-400">Sources</span>
              </div>
              <div className="space-y-2">
                {sources.map((source, idx) => (
                  <div key={idx} className="text-sm text-gray-300">
                    <span className="font-medium">{source.title || `Source ${idx + 1}`}</span>
                    {source.metadata && (
                      <span className="text-gray-400 ml-2">
                        - {source.metadata.section || source.metadata.source || ''}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-black/30 backdrop-blur-md border-t border-purple-500/20 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="relative flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about family law..."
                disabled={isLoading}
                rows={1}
                className="flex-1 px-4 py-3 bg-white/10 backdrop-blur-sm border border-purple-500/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
              <button
                onClick={() => sendMessage()}
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2 text-center">
              Powered by Family Law RAG System • Press Enter to send
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LegalAssistChat;
