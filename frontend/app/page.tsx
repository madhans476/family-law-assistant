"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Plus, Scale, BookOpen, Loader2, MessageSquare, ExternalLink, FileText, AlertCircle, CheckCircle, Clock, Info } from 'lucide-react';

// Types
interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
  messageType?: 'clarification' | 'information_gathering' | 'final_response';
  infoCollected?: Record<string, string>;
  infoNeeded?: string[];
}

interface Source {
  title: string;
  url?: string;
  category?: string;
}

interface Conversation {
  conversation_id: string;
  last_modified: string;
  message_count: number;
  status: 'analyzing' | 'gathering_info' | 'completed';
  user_intent: string;
}

const LegalAssistChat: React.FC = () => {
  const [threads, setThreads] = useState<Conversation[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [currentMessageType, setCurrentMessageType] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [conversationStatus, setConversationStatus] = useState<string>('ready');
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
    setMessages([]);
    setSources([]);
    setStreamingMessage('');
    setError(null);
    setInput('');
    setConversationStatus('ready');
  };

  const loadThread = async (threadId: string) => {
    try {
      setError(null);
      setStreamingMessage('');
      const res = await fetch(`${API_BASE}/history/${threadId}`);
      if (!res.ok) throw new Error('Failed to load conversation');
      const data = await res.json();
      setCurrentThreadId(threadId);
      
      const formattedMessages: Message[] = data.messages.map((msg: any) => ({
        role: msg.role === 'HumanMessage' ? 'user' : 'assistant',
        content: msg.content,
        timestamp: new Date()
      }));
      setMessages(formattedMessages);
      setSources([]);
      
      // Set status based on state
      const state = data.state || {};
      if (state.has_sufficient_info) {
        setConversationStatus('completed');
      } else if (state.in_gathering_phase) {
        setConversationStatus('gathering_info');
      } else {
        setConversationStatus('analyzing');
      }
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
        setConversationStatus('ready');
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
    setSources([]);
    setError(null);
    setCurrentMessageType(null);

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
      let messageType: string | null = null;
      let infoCollected: Record<string, string> = {};
      let infoNeeded: string[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'clarification') {
                // Clarification question received
                accumulatedText = data.content;
                messageType = 'clarification';
                setStreamingMessage(accumulatedText);
                setConversationStatus('clarifying');
              } else if (data.type === 'information_gathering') {
                // Information gathering question
                accumulatedText = data.content;
                messageType = 'information_gathering';
                infoCollected = data.info_collected || {};
                infoNeeded = data.info_needed || [];
                setStreamingMessage(accumulatedText);
                setConversationStatus('gathering_info');
              } else if (data.type === 'token') {
                // Streaming final response
                accumulatedText += data.content;
                messageType = 'final_response';
                setStreamingMessage(accumulatedText);
                setConversationStatus('generating');
              } else if (data.type === 'sources') {
                setSources(data.sources || []);
              } else if (data.type === 'done') {
                messageType = data.message_type || 'final_response';
                if (accumulatedText) {
                  const newMessage: Message = { 
                    role: 'assistant', 
                    content: accumulatedText,
                    timestamp: new Date(),
                    messageType: messageType as any
                  };
                  
                  if (messageType === 'information_gathering') {
                    newMessage.infoCollected = data.info_collected;
                    newMessage.infoNeeded = data.info_needed;
                  }
                  
                  setMessages(prev => [...prev, newMessage]);
                }
                setStreamingMessage('');
                
                // Update status
                if (messageType === 'final_response') {
                  setConversationStatus('completed');
                } else if (messageType === 'information_gathering') {
                  setConversationStatus('gathering_info');
                } else {
                  setConversationStatus('clarifying');
                }
              } else if (data.type === 'error') {
                throw new Error(data.message);
              }
            } catch (parseErr) {
              console.warn('Parse error:', parseErr);
            }
          }
        }
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
    
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/```(.*?)```/gs, '<pre class="bg-gray-800 p-3 rounded-lg my-2 overflow-x-auto"><code>$1</code></pre>');
    text = text.replace(/`([^`]+)`/g, '<code class="bg-gray-800 px-2 py-1 rounded text-sm">$1</code>');
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

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'analyzing': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'gathering_info': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 'completed': return 'bg-green-500/20 text-green-300 border-green-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'analyzing': return <Loader2 className="w-4 h-4 animate-spin" />;
      case 'gathering_info': return <Info className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const getStatusText = (status: string): string => {
    switch (status) {
      case 'analyzing': return 'Analyzing';
      case 'gathering_info': return 'Gathering Info';
      case 'completed': return 'Completed';
      default: return 'Ready';
    }
  };

  const renderInfoProgress = (msg: Message) => {
    if (msg.messageType !== 'information_gathering') return null;

    const collected = Object.keys(msg.infoCollected || {}).length;
    const needed = (msg.infoNeeded || []).length;
    const total = collected + needed;

    if (total === 0) return null;

    return (
      <div className="mt-3 p-3 bg-slate-800/30 rounded-lg border border-blue-500/20">
        <div className="flex items-center gap-2 mb-2">
          <Info className="w-4 h-4 text-blue-400" />
          <span className="text-sm font-medium text-blue-300">Information Collection Progress</span>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-400">
            <span>Collected: {collected}</span>
            <span>Remaining: {needed}</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(collected / total) * 100}%` }}
            />
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 overflow-hidden">
      {/* Sidebar */}
      <div className="w-80 bg-slate-950/50 backdrop-blur-xl border-r border-blue-500/20 flex flex-col shadow-2xl">
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

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {threads.length === 0 ? (
            <div className="text-center text-gray-400 py-12 px-4">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No conversations yet</p>
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
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="w-4 h-4 text-blue-400 flex-shrink-0" />
                      <p className="text-white font-medium text-sm truncate">
                        {thread.user_intent || 'New Case'}
                      </p>
                    </div>
                    <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs mb-2 ${getStatusColor(thread.status)}`}>
                      {getStatusIcon(thread.status)}
                      <span>{getStatusText(thread.status)}</span>
                    </div>
                    <p className="text-gray-400 text-xs">
                      {new Date(thread.last_modified).toLocaleDateString('en-IN', {
                        day: 'numeric',
                        month: 'short'
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
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="bg-slate-950/50 backdrop-blur-xl border-b border-blue-500/20 p-6 shadow-xl">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl">
              <BookOpen className="w-6 h-6 text-blue-400" />
            </div>
            <div className="flex-1">
              <h2 className="text-white font-semibold text-lg">AI Legal Consultation</h2>
              <p className="text-sm text-gray-400 mt-1">
                Intelligent information gathering • Step-by-step guidance
              </p>
            </div>
            {currentThreadId && (
              <div className={`px-4 py-2 rounded-lg border ${getStatusColor(conversationStatus)}`}>
                <div className="flex items-center gap-2">
                  {getStatusIcon(conversationStatus)}
                  <p className="text-xs font-medium">{getStatusText(conversationStatus)}</p>
                </div>
              </div>
            )}
          </div>
        </div>

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
                  I'll guide you through an intelligent conversation to understand your situation before providing legal advice.
                </p>
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
                    : msg.messageType === 'clarification'
                    ? 'bg-yellow-500/10 border-2 border-yellow-500/30 text-gray-100'
                    : msg.messageType === 'information_gathering'
                    ? 'bg-blue-500/10 border-2 border-blue-500/30 text-gray-100'
                    : 'bg-slate-800/50 backdrop-blur-sm text-gray-100 border border-blue-500/10'
                } rounded-2xl shadow-lg overflow-hidden`}
              >
                <div className="px-6 py-4">
                  {msg.messageType === 'clarification' && (
                    <div className="flex items-center gap-2 mb-2 pb-2 border-b border-yellow-500/20">
                      <AlertCircle className="w-4 h-4 text-yellow-400" />
                      <span className="text-xs font-semibold text-yellow-300 uppercase">Need Clarification</span>
                    </div>
                  )}
                  {msg.messageType === 'information_gathering' && (
                    <div className="flex items-center gap-2 mb-2 pb-2 border-b border-blue-500/20">
                      <Info className="w-4 h-4 text-blue-400" />
                      <span className="text-xs font-semibold text-blue-300 uppercase">Gathering Information</span>
                    </div>
                  )}
                  <div
                    dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                    className="prose prose-invert max-w-none"
                  />
                  {renderInfoProgress(msg)}
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
                  <div key={idx} className="bg-slate-800/30 rounded-lg p-4">
                    {source.url ? (
                      <a 
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-300 hover:text-blue-200 font-medium flex items-center gap-2 group"
                      >
                        <span>{source.title}</span>
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    ) : (
                      <span className="text-gray-200 font-medium">{source.title}</span>
                    )}
                    {source.category && (
                      <span className="inline-block mt-2 px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full">
                        {source.category}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

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
                className="flex-1 px-5 py-4 bg-slate-800/50 backdrop-blur-sm border border-blue-500/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none disabled:opacity-50"
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
              <button
                onClick={() => sendMessage()}
                disabled={isLoading || !input.trim()}
                className="px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-xl hover:from-blue-700 hover:to-blue-600 disabled:opacity-50 transition-all shadow-lg font-medium flex items-center gap-2"
              >
                {isLoading ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /><span className="hidden sm:inline">Processing</span></>
                ) : (
                  <><Send className="w-5 h-5" /><span className="hidden sm:inline">Send</span></>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default LegalAssistChat;