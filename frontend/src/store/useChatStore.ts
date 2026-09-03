import { create } from "zustand"
import type { ChatMessage } from "@/types/diagnosis"

interface ChatState {
  messages: ChatMessage[]
  isStreaming: boolean
  addMessage: (msg: Omit<ChatMessage, "id" | "timestamp">) => string
  updateMessageContent: (id: string, content: string) => void
  appendToMessage: (id: string, token: string) => void
  setIsStreaming: (streaming: boolean) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isStreaming: false,
  addMessage: (msg) => {
    const id = Math.random().toString(36).substring(2, 9)
    const newMsg: ChatMessage = {
      ...msg,
      id,
      timestamp: Date.now(),
    }
    set((state) => ({ messages: [...state.messages, newMsg] }))
    return id
  },
  updateMessageContent: (id, content) =>
    set((state) => ({
      messages: state.messages.map((m) => (m.id === id ? { ...m, content } : m)),
    })),
  appendToMessage: (id, token) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      ),
    })),
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  clearMessages: () => set({ messages: [] }),
}))
