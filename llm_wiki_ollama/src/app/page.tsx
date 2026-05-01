"use client";

import { useState, useRef, useEffect } from "react";
import { UploadCloud, FileText, Send, Loader2, Database, Cpu, Search, CheckCircle2, Trash2 } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [contextText, setContextText] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const [query, setQuery] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([]);

  const [models, setModels] = useState<{ name: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [wikiFiles, setWikiFiles] = useState<string[]>([]);
  const [wikiDocuments, setWikiDocuments] = useState<{title: string, content: string}[]>([]);

  useEffect(() => {
    // Fetch models
    fetch("/api/models")
      .then((res) => res.json())
      .then((data) => {
        if (data.models && data.models.length > 0) {
          setModels(data.models);
          const gemmaModel = data.models.find((m: any) => m.name.includes("gemma3"));
          setSelectedModel(gemmaModel ? gemmaModel.name : data.models[0].name);
        }
      })
      .catch((err) => console.error("Error fetching models:", err));
      
    // Fetch existing wiki
    fetchWiki();
  }, []);

  const fetchWiki = async () => {
    try {
      const res = await fetch("/api/wiki");
      const data = await res.json();
      
      setContextText(data.text || "");
      setWikiFiles(data.files || []);
      setWikiDocuments(data.documents || []);
      
      if (data.text) {
        setUploadSuccess(true);
      } else {
        setUploadSuccess(false);
      }
    } catch (err) {
      console.error("Error fetching wiki:", err);
    }
  };

  const handleDeleteWiki = async (filename: string) => {
    if (!confirm(`Are you sure you want to remove ${filename.replace(/\.md$/, ".pdf")} from the knowledge base?`)) return;
    try {
      const res = await fetch(`/api/wiki?file=${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        await fetchWiki();
      } else {
        const data = await res.json();
        alert(data.error || "Failed to delete file");
      }
    } catch (error) {
      console.error("Error deleting file:", error);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Auto upload & ingest
      await uploadFile(selectedFile);
    }
  };

  const uploadFile = async (selectedFile: File) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", selectedModel);

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadSuccess(true);
        // Refresh the wiki context
        await fetchWiki();
      } else {
        alert(data.error || "Failed to ingest PDF");
      }
    } catch (error) {
      console.error(error);
      alert("Error uploading file");
    } finally {
      setIsUploading(false);
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userQuery = query.trim();
    setMessages((prev) => [...prev, { role: "user", content: userQuery }]);
    setQuery("");
    setIsQuerying(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userQuery,
          model: selectedModel,
        }),
      });

      const data = await res.json();
      if (res.ok) {
        setMessages((prev) => [...prev, { role: "assistant", content: data.response }]);
      } else {
        setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${data.error}` }]);
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, { role: "assistant", content: "Error communicating with server." }]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-zinc-900 font-sans selection:bg-indigo-500/30">
      {/* Background gradients */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-500/10 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-purple-500/10 blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-[1400px] mx-auto p-6 flex flex-col h-screen">
        
        {/* Header */}
        <header className="flex items-center justify-between py-6 mb-4 border-b border-zinc-200">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-indigo-50 rounded-xl border border-indigo-100">
              <Database className="w-6 h-6 text-indigo-600" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-zinc-800">Ollama LLM-Wiki</h1>
              <p className="text-xs text-zinc-500 font-medium">Local Knowledge Engine</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white rounded-lg border border-zinc-200 text-sm font-medium shadow-sm">
              <Cpu className="w-4 h-4 text-emerald-500" />
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="bg-transparent border-none outline-none text-zinc-700 focus:ring-0 cursor-pointer"
              >
                {models.length > 0 ? (
                  models.map((m) => (
                    <option key={m.name} value={m.name} className="bg-white">{m.name}</option>
                  ))
                ) : (
                  <option value="" className="bg-white">No models found</option>
                )}
              </select>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-col md:flex-row gap-6 flex-1 min-h-0">
          
          {/* Left Sidebar - Upload */}
          <div className="w-full md:w-72 flex flex-col gap-4 shrink-0">
            <div className="bg-white border border-zinc-200 rounded-2xl p-5 flex flex-col h-full shadow-sm">
              <h2 className="text-sm font-semibold text-zinc-800 mb-4 flex items-center gap-2 uppercase tracking-wider">
                <FileText className="w-4 h-4 text-zinc-500" />
                Knowledge Base
              </h2>
              
              <div 
                onClick={() => fileInputRef.current?.click()}
                className={`relative group flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
                  uploadSuccess 
                    ? "border-emerald-500/30 bg-emerald-50 hover:bg-emerald-100" 
                    : isUploading
                    ? "border-indigo-500/30 bg-indigo-50"
                    : "border-zinc-300 hover:border-indigo-400 hover:bg-zinc-50"
                }`}
              >
                <input
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                
                {isUploading ? (
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
                    <p className="text-sm text-zinc-600 font-medium">Extracting text...</p>
                  </div>
                ) : uploadSuccess ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="p-3 bg-emerald-100 rounded-full">
                      <CheckCircle2 className="w-8 h-8 text-emerald-600" />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-semibold text-emerald-600">PDF Loaded</p>
                      <p className="text-xs text-zinc-500 mt-1 truncate max-w-[180px]">{file?.name}</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <div className="p-3 bg-zinc-100 rounded-full group-hover:scale-110 group-hover:bg-indigo-100 transition-all duration-300">
                      <UploadCloud className="w-8 h-8 text-zinc-500 group-hover:text-indigo-500" />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-zinc-700">Click to upload PDF</p>
                      <p className="text-xs text-zinc-500 mt-1">Extracts text for context</p>
                    </div>
                  </div>
                )}
              </div>

              {wikiFiles.length > 0 && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Processed Documents</p>
                  <ul className="space-y-1.5">
                    {wikiFiles.map((file, idx) => (
                      <li key={idx} className="flex items-center justify-between gap-2 text-xs text-zinc-700 bg-zinc-50 p-2 rounded-lg border border-zinc-200 group transition-all hover:border-indigo-200">
                        <div className="flex items-center gap-2 overflow-hidden">
                          <FileText className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                          <span className="truncate" title={file.replace(/\.md$/, ".pdf")}>{file.replace(/\.md$/, ".pdf")}</span>
                        </div>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteWiki(file);
                          }}
                          className="p-1 text-zinc-400 hover:text-red-600 hover:bg-red-50 rounded opacity-0 group-hover:opacity-100 transition-all shrink-0"
                          title="Delete from knowledge base"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}


            </div>
          </div>

          {/* Right Section - Chat */}
          <div className="flex-1 flex flex-col bg-white border border-zinc-200 rounded-2xl shadow-sm overflow-hidden relative">
            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center opacity-50">
                  <Search className="w-12 h-12 text-zinc-400 mb-4" />
                  <h3 className="text-lg font-medium text-zinc-800 mb-2">Ready to explore</h3>
                  <p className="text-sm text-zinc-500 max-w-sm">
                    Upload a PDF to build the context, then ask any question. The local LLM will use the document to formulate an answer.
                  </p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div 
                      className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                        msg.role === "user" 
                          ? "bg-indigo-600 border border-indigo-700 text-white shadow-sm" 
                          : "bg-zinc-50 border border-zinc-200 text-zinc-800 leading-relaxed shadow-sm"
                      }`}
                    >
                      {msg.role === "assistant" && (
                        <div className="flex items-center gap-2 mb-2 pb-2 border-b border-zinc-200">
                          <Cpu className="w-3.5 h-3.5 text-emerald-600" />
                          <span className="text-xs font-medium text-zinc-500">{selectedModel}</span>
                        </div>
                      )}
                      <div className="text-[15px]">
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({node, ...props}) => <p className="mb-3 last:mb-0 leading-relaxed" {...props} />,
                            ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
                            ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3 space-y-1" {...props} />,
                            h1: ({node, ...props}) => <h1 className="text-lg font-bold mt-4 mb-2" {...props} />,
                            h2: ({node, ...props}) => <h2 className="text-base font-bold mt-4 mb-2" {...props} />,
                            h3: ({node, ...props}) => <h3 className="text-sm font-semibold mt-3 mb-1" {...props} />,
                            code: ({node, inline, ...props}: any) => 
                              inline 
                                ? <code className="bg-zinc-200 px-1.5 py-0.5 rounded-md text-[13px] font-mono text-zinc-800" {...props} />
                                : <pre className="bg-white border border-zinc-200 p-4 rounded-xl overflow-x-auto text-[13px] font-mono my-3 shadow-inner text-zinc-800"><code {...props} /></pre>,
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ))
              )}
              {isQuerying && (
                <div className="flex justify-start">
                  <div className="bg-zinc-50 border border-zinc-200 rounded-2xl px-5 py-4 flex items-center gap-3 shadow-sm">
                    <Loader2 className="w-4 h-4 text-indigo-500 animate-spin" />
                    <span className="text-sm text-zinc-600">Synthesizing answer...</span>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="p-4 bg-zinc-50 border-t border-zinc-200">
              <form onSubmit={handleQuery} className="relative flex items-center">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={uploadSuccess ? "Ask a question about your document..." : "Upload a PDF first to ask questions..."}
                  disabled={isQuerying || !uploadSuccess}
                  className="w-full bg-white border border-zinc-300 text-zinc-900 placeholder-zinc-400 rounded-xl pl-4 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                />
                <button 
                  type="submit"
                  disabled={!query.trim() || isQuerying || !uploadSuccess}
                  className="absolute right-2 p-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-200 disabled:text-zinc-400 disabled:cursor-not-allowed text-white rounded-lg transition-colors shadow-sm"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </div>

          {/* Right Sidebar - Wiki Preview */}
          {contextText && (
            <div className="w-full md:w-80 flex flex-col gap-4 shrink-0">
              <div className="bg-white border border-zinc-200 rounded-2xl p-5 flex flex-col h-full shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold text-zinc-800 flex items-center gap-2 uppercase tracking-wider">
                    Wiki Preview
                  </h2>
                  <span className="text-[10px] bg-zinc-100 border border-zinc-200 text-zinc-500 px-2 py-0.5 rounded-full">
                    {contextText.length.toLocaleString()} chars
                  </span>
                </div>
                <div className="flex-1 bg-zinc-50 border border-zinc-200 rounded-xl p-4 overflow-y-auto custom-scrollbar space-y-6">
                  {wikiDocuments.map((doc, idx) => (
                    <div key={idx} className="pb-6 border-b border-zinc-200 last:border-0 last:pb-0">
                      <div className="mb-3">
                        <span className="text-[10px] font-semibold text-indigo-700 bg-indigo-100 px-2.5 py-1 rounded-md border border-indigo-200">
                          {doc.title}
                        </span>
                      </div>
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          h1: ({node, ...props}) => <h1 className="text-sm font-bold text-zinc-900 mb-3 hidden" {...props} />, // Hide top level h1 to save space
                          h2: ({node, ...props}) => <h2 className="text-xs font-bold text-zinc-800 mt-4 mb-2" {...props} />,
                          h3: ({node, ...props}) => <h3 className="text-[11px] font-semibold text-indigo-600 mt-3 mb-1 uppercase tracking-wider" {...props} />,
                          p: ({node, ...props}) => <p className="text-xs text-zinc-600 leading-relaxed mb-3" {...props} />,
                          ul: ({node, ...props}) => <ul className="list-disc pl-4 text-xs text-zinc-600 mb-3 space-y-1" {...props} />,
                          ol: ({node, ...props}) => <ol className="list-decimal pl-4 text-xs text-zinc-600 mb-3 space-y-1" {...props} />,
                          li: ({node, ...props}) => <li className="" {...props} />,
                          strong: ({node, ...props}) => <strong className="text-zinc-800 font-semibold" {...props} />,
                          code: ({node, inline, ...props}: any) => 
                            inline 
                              ? <code className="bg-zinc-200 px-1 py-0.5 rounded text-[10px] font-mono text-zinc-700" {...props} />
                              : <pre className="bg-white p-2 rounded-lg overflow-x-auto text-[10px] font-mono my-2 border border-zinc-200 shadow-sm"><code {...props} /></pre>,
                        }}
                      >
                        {doc.content.length > 300 
                          ? doc.content.slice(0, 300) + "...\n\n_*(Preview truncated)*_"
                          : doc.content}
                      </ReactMarkdown>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
