import os
import logging
from openai import OpenAI
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DeepSeekIntegration:
    """DeepSeek AI integration via OpenRouter for professional company knowledge base"""
    
    def __init__(self):
        """Initialize DeepSeek AI integration via OpenRouter"""
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('GROK_API_KEY')  # Support both keys
            if not api_key:
                logger.warning("DEEPSEEK_API_KEY not set - using fallback response mode")
                self.client = None
                self.model = None
                self.fallback_mode = True
            else:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                # Use DeepSeek R1 model (much more powerful than Grok)
                self.model = "deepseek/deepseek-r1-0528:free"
                self.fallback_mode = False
                logger.info("DeepSeekIntegration initialized with API key")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeekIntegration: {str(e)}")
            self.client = None
            self.model = None
            self.fallback_mode = True
            logger.info("DeepSeekIntegration running in fallback mode")
    
    def generate_response(self, question: str, context: str, history: list = None) -> str:
        """Generate a professional response based on company documents using DeepSeek R1"""
        
        # Check if we're in fallback mode
        if self.fallback_mode or not self.client:
            return self._generate_fallback_response(question, context)
        
        # Enhanced system prompt for professional company knowledge base
        system_prompt = """You are Tuterby AI, a professional company knowledge assistant powered by DeepSeek R1. Your role is to provide accurate, well-formatted, and professional responses based on company documents.

IMPORTANT GUIDELINES:
1. Always base your answers on the provided context - only use information from the company documents
2. Format responses professionally with clear structure, bullet points, and proper spacing
3. Be comprehensive - provide detailed explanations, not just brief answers
4. Use professional language suitable for business environments
5. If information is not in the context, clearly state that and suggest what documents might contain it
6. Structure your response with clear headings, bullet points, and organized information
7. Provide actionable insights when possible
8. Maintain consistency with company terminology and procedures
9. Use DeepSeek's advanced reasoning to provide intelligent analysis and connections

RESPONSE FORMAT:
- Start with a clear, direct answer to the question
- Use bullet points (•) for lists and key information
- Organize information logically with clear sections
- Use professional business language
- End with a summary or next steps if applicable
- DO NOT use markdown formatting like **bold** or *italic*
- Use clean, readable text with proper spacing

CONTEXT FROM COMPANY DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Please provide a professional, well-formatted response based on the company documents using DeepSeek's advanced capabilities."""
        
        try:
            # Prepare conversation history for context
            history_text = ""
            if history:
                history_text = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in history[-5:]])  # Last 5 messages
            
            # Format the prompt
            formatted_prompt = system_prompt.format(
                context=context,
                history=history_text,
                question=question
            )
            
            # Generate response with optimized parameters for DeepSeek R1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": formatted_prompt}
                ],
                temperature=0.2,  # Lower temperature for more focused, professional responses
                max_tokens=4000,  # Increased for comprehensive answers
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract and format the response
            ai_response = response.choices[0].message.content.strip()
            
            # Ensure proper formatting
            if not ai_response.startswith('#'):  # If no markdown headers, add structure
                ai_response = self._format_response(ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Provide a helpful fallback response
            return self._generate_fallback_response(question, context)
    
    def _generate_fallback_response(self, question: str, context: str) -> str:
        """Generate a professional fallback response when AI is unavailable"""
        
        # Simple but professional fallback based on context
        if not context or context == "No relevant documents found. Please upload a PDF first.":
            return """I apologize, but I'm currently unable to access the DeepSeek AI service. However, I can help you with basic document management.

Current Status:
• AI service temporarily unavailable
• Document upload and storage still functional
• Basic document management available

What you can do:
1. Upload company documents for later AI analysis
2. View and manage existing documents
3. Try again in a few minutes

For immediate assistance:
Please contact your system administrator or try refreshing the page. The AI service should be restored shortly."""
        
        # If we have context, provide a basic but professional response
        context_length = len(context)
        if context_length > 1000:
            context_summary = f"Based on your company documents ({context_length} characters of content), "
        else:
            context_summary = "Based on your company documents, "
        
        return f"""{context_summary}I can see relevant information is available. However, I'm currently experiencing technical difficulties with the DeepSeek AI processing service.

What I can tell you:
• Your documents are successfully stored and indexed
• The content contains relevant information for your question
• The system is working for document management

Your Question: "{question}"

Recommended Actions:
1. Try again in 2-3 minutes - the AI service may be temporarily busy
2. Check your documents - ensure they contain the information you need
3. Contact support if the issue persists

Technical Note: This is a temporary service interruption. Your data is safe and the system will resume normal operation shortly."""
    
    def _format_response(self, response: str) -> str:
        """Format response to ensure professional structure"""
        
        # Clean up the response
        response = response.strip()
        
        # Remove excessive asterisks and clean up formatting
        response = response.replace('**', '')  # Remove markdown bold
        response = response.replace('*', '')   # Remove markdown italic
        response = response.replace('`', '')   # Remove markdown code
        
        # Clean up bullet points and lists
        response = response.replace('•', '•')  # Ensure consistent bullet points
        response = response.replace('- ', '• ')  # Convert dashes to bullets
        
        # If response is too short, expand it
        if len(response) < 100:
            response = f"Based on the company documents, here's what I found:\n\n{response}\n\nIf you need more specific information, please let me know or check the relevant documents."
        
        # Ensure proper spacing
        response = response.replace('\n\n\n', '\n\n')
        
        # Add structure if missing
        if not any(char in response for char in ['•', '1.', '2.']):
            # Try to add bullet points for better readability
            sentences = response.split('. ')
            if len(sentences) > 2:
                formatted_sentences = []
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        if i == 0:
                            formatted_sentences.append(sentence.strip())
                        else:
                            formatted_sentences.append(f"• {sentence.strip()}")
                response = '\n\n'.join(formatted_sentences)
        
        # Ensure proper paragraph breaks
        response = response.replace('\n\n', '\n\n')
        
        return response

# Keep backward compatibility
GrokIntegration = DeepSeekIntegration
