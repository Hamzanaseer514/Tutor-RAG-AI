import os
import openai
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class GrokIntegration:
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        api_key = api_key or os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "x-ai/grok-3"
        logger.info("GrokIntegration initialized")
    
    def generate_response(self, question: str, context: str, history: List[Dict] = None) -> str:
        """Generate a response using Grok AI with context and conversation history"""
        try:
            # Prepare system prompt with enhanced instructions
            system_prompt = self._create_system_prompt(context)
            
            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (last 5 exchanges to maintain context)
            if history:
                # Take last 10 messages (5 exchanges) to maintain conversation flow
                recent_history = history[-10:]
                messages.extend(recent_history)
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            logger.info(f"Generating response with {len(messages)} messages")
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=1500,  # Increased for more detailed responses
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"Generated response: {len(ai_response)} characters")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error calling Grok API: {str(e)}")
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"
    
    def _create_system_prompt(self, context: str) -> str:
        """Create a comprehensive system prompt for better responses"""
        return f"""You are Tuterby AI, a professional and knowledgeable AI assistant specialized in analyzing and answering questions about PDF documents.

IMPORTANT INSTRUCTIONS:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain enough information to answer a question, say so clearly
3. Provide accurate, well-structured responses based on the document content
4. If asked about something not in the document, politely explain that you can only answer based on the uploaded document
5. Be professional, helpful, and concise
6. Support both English and Urdu languages in your responses
7. When referencing information from the document, be specific about what you found

CONTEXT FROM DOCUMENT:
{context}

RESPONSE GUIDELINES:
- Answer in the same language as the question (English or Urdu)
- Provide specific references to the document when possible
- If the question is unclear, ask for clarification
- Be helpful but don't make up information not present in the document
- Structure your response logically and clearly

Remember: You are an expert at analyzing this specific document. Stick to the facts and information provided in the context above."""
    
    def generate_summary(self, context: str) -> str:
        """Generate a summary of the document content"""
        try:
            system_prompt = """You are Tuterby AI. Create a concise, professional summary of the provided document content. 
            Focus on the main topics, key points, and overall structure. Keep it under 200 words."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please provide a summary of this document:\n\n{context}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary at this time."
    
    def answer_follow_up(self, question: str, context: str, conversation_history: List[Dict]) -> str:
        """Handle follow-up questions with conversation context"""
        try:
            # Create a more focused system prompt for follow-up questions
            system_prompt = f"""You are Tuterby AI. This is a follow-up question in an ongoing conversation about a document.

Previous conversation context has been provided. Use both the document content and conversation history to provide a coherent response.

Document Context:
{context}

Remember to:
- Reference previous parts of the conversation when relevant
- Maintain consistency with your previous answers
- Build upon the established context
- Be conversational but professional"""

            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_tokens=1200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error handling follow-up question: {str(e)}")
            return f"I apologize, but I'm having trouble processing your follow-up question. Error: {str(e)}"
    
    def validate_question(self, question: str) -> Dict[str, any]:
        """Validate if a question is appropriate for document analysis"""
        try:
            system_prompt = """Analyze if this question is appropriate for document analysis. 
            Return a JSON response with:
            - is_valid: boolean
            - reason: string explaining why it's valid/invalid
            - suggested_rephrasing: string if the question can be improved"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )
            
            # Try to parse JSON response
            try:
                import json
                result = json.loads(response.choices[0].message.content)
                return result
            except:
                # Fallback if JSON parsing fails
                return {
                    "is_valid": True,
                    "reason": "Question appears appropriate for document analysis",
                    "suggested_rephrasing": question
                }
                
        except Exception as e:
            logger.error(f"Error validating question: {str(e)}")
            return {"is_valid": True, "reason": "Validation failed", "suggested_rephrasing": question}
