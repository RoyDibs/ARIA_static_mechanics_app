from openai import OpenAI
from typing import List, Dict, Any
import json
from pathlib import Path
import os
from .embedding.rag_retriever import StaticsMechanicsRAG
from .database.conversation_storage import conversation_storage
import logging
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StaticsMechanicsTA:
    def __init__(self, base_path: str, api_key: str):
        self.base_path = Path(base_path)
        # Use e5-large-v2 for query embeddings to match deployed embeddings
        self.rag = StaticsMechanicsRAG(base_path, model_name="intfloat/e5-large-v2")
        self.client = OpenAI(api_key=api_key)
        self.session_id = str(uuid.uuid4())  # Generate unique session ID
        
        # Enhanced system prompt for strict academic tutor with LaTeX formatting
        self.system_prompt = """
You are ARIA, a strict Academic Tutor for Statics & Mechanics of Materials. You EXCLUSIVELY respond to questions related to statics and mechanics. For any other topics, respond exactly with: "This question is not relevant to the course."

CORE PRINCIPLES:
1. Provide clear, direct explanations of concepts
2. Deliver step-by-step solutions for numerical problems
3. Include relevant formulas with variable definitions
4. Maintain professional academic tone
5. Focus solely on statics and mechanics content
6. Give complete information without unnecessary elaboration
7. Use precise technical language
8. Format ALL mathematical expressions using LaTeX notation

MATHEMATICAL FORMATTING REQUIREMENTS:
- Use $expression$ for inline mathematical expressions
- Use $$expression$$ for display equations
- Format all numerical values, variables, and units in LaTeX
- Examples: $\sigma = \frac{F}{A}$, $E = 200 \text{ GPa}$, $\theta = 45°$
- Include proper subscripts, superscripts, and Greek letters

RESPONSE STRUCTURE:
- Concept Explanation: Clear theory and fundamental principles with LaTeX formatting
- Solution Steps: Direct problem-solving approach with calculations in LaTeX
- Formulas: Relevant equations with variable definitions in proper LaTeX format

Remember: You are a strict academic tutor focused exclusively on statics and mechanics content. Reject any non-course questions with the specified response. Always use LaTeX for mathematical notation.
"""
    
    def generate_response(
        self, 
        student_question: str, 
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate a teaching-focused response"""
        
        start_time = datetime.now()
        
        try:
            # Check if question is relevant to course materials with validation
            is_course_relevant = self._validate_and_check_course_relevance(student_question)
            logger.info(f"Question relevance to course materials: {is_course_relevant}")
            
            if is_course_relevant:
                # For course-relevant questions: retrieve content and use course context
                relevant_content = self._get_relevant_content(student_question)
                context = self._prepare_context(relevant_content)
                messages = self._create_messages(
                    student_question, 
                    context, 
                    conversation_history
                )
            else:
                # For irrelevant questions: use general assistant mode without course context
                relevant_content = []  # No course content for irrelevant questions
                messages = self._create_general_messages(
                    student_question,
                    conversation_history
                )
            
            # Generate response with GPT-4 using parameters for direct academic instruction
            response = self.client.chat.completions.create(
                model="gpt-4", 
                messages=messages,
                max_tokens=600,  # Focused on direct instruction
                temperature=0.3,  # Lower for consistent academic responses
                presence_penalty=0.1,  # Light penalty for focused responses
                frequency_penalty=0.1,  # Light penalty for concise instruction
                top_p=0.9  # Focused response generation
            )
            
            assistant_response = response.choices[0].message.content
            logger.info(f"Generated answer length: {len(assistant_response)}")
            
            # Enhanced validation and error handling for source addition
            try:
                # CRITICAL: Never add sources if the response indicates the question is not relevant
                if assistant_response and "This question is not relevant to the course" in assistant_response:
                    logger.info("Response indicates question is not course-relevant - explicitly skipping all source references")
                
                elif is_course_relevant and relevant_content:
                    # Validate that relevant_content is properly formatted
                    if not isinstance(relevant_content, list):
                        logger.error(f"Invalid relevant_content type: {type(relevant_content)}")
                        relevant_content = []
                    
                    # Filter out invalid content entries
                    valid_content = []
                    for content in relevant_content:
                        if isinstance(content, dict) and "metadata" in content:
                            valid_content.append(content)
                        else:
                            logger.warning(f"Skipping invalid content entry: {type(content)}")
                    
                    if valid_content:
                        source_references = self._format_source_references(valid_content)
                        logger.info(f"Source references generated: {source_references}")
                        
                        # Additional validation of source references format
                        if source_references and isinstance(source_references, str) and source_references.strip():
                            # Double-check that response is still course-relevant before adding sources
                            if "This question is not relevant to the course" not in assistant_response:
                                # Enhanced source display format
                                if ";" in source_references:
                                    # Multiple sources - format as detailed list
                                    assistant_response += f"\n\n📚 **Sources Referenced:**\n{source_references}"
                                else:
                                    # Single source - format inline
                                    assistant_response += f"\n\n📚 **Source:** {source_references}"
                                logger.info(f"Enhanced sources successfully added to response")
                            else:
                                logger.info("Response contains 'not relevant' message - sources explicitly excluded")
                        else:
                            logger.warning("Source references formatting failed or returned empty")
                    else:
                        logger.warning("No valid content entries found for source generation")
                        
                elif is_course_relevant and not relevant_content:
                    logger.info("Question is course-relevant but no content retrieved - skipping sources")
                    
                elif not is_course_relevant and relevant_content:
                    # This shouldn't happen, but handle gracefully
                    logger.warning("Content retrieved for non-course question - validation logic may need review")
                    # Explicitly skip sources for non-course questions
                    
                else:
                    logger.info("Question not course-relevant and no content found - correctly skipping source references")
                    
            except Exception as source_error:
                logger.error(f"Error in source reference processing: {source_error}")
                # Continue without sources rather than failing the entire response
                logger.info("Continuing without source references due to error")
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Log the interaction to Supabase
            self._log_interaction(
                student_question, 
                assistant_response, 
                relevant_content, 
                response_time
            )
            
            return {
                "response": assistant_response,
                "relevant_topics": [content["metadata"].get("topic", "Unknown") for content in relevant_content] if relevant_content else [],
                "concepts_covered": self._extract_concepts_from_content(relevant_content) if relevant_content else [],
                "suggested_review": self._suggest_review_materials(relevant_content) if relevant_content else [],
                "is_course_relevant": is_course_relevant,
                "context_sources": [c["metadata"].get("source_file", "unknown") for c in relevant_content] if relevant_content else []
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm experiencing technical difficulties processing your question. Please rephrase your question clearly. If it's related to statics and mechanics of materials, I'll provide direct instruction. If not, this question is not relevant to the course.",
                "error": str(e)
            }
    
    def _get_relevant_content(self, question: str) -> List[Dict]:
        """Retrieve relevant content based on question type"""
        
        # Get concept-related content
        concept_content = self.rag.get_concept_related_content(question, n_results=2)
        
        # Get similar exercises
        exercise_content = self.rag.get_similar_exercises(question, n_results=2)
        
        # Get solution guidance
        solution_content = self.rag.get_solution_guidance(question, n_results=1)
        
        # Combine and deduplicate
        all_content = concept_content + exercise_content + solution_content
        
        # Remove duplicates based on content similarity
        unique_content = self._deduplicate_content(all_content)
        
        return unique_content[:5]  # Limit to top 5 most relevant
    
    def _prepare_context(self, relevant_content: List[Dict]) -> str:
        """Prepare context from retrieved content"""
        context_parts = []
        
        for content in relevant_content:
            content_type = content["metadata"].get("content_type", "unknown")
            topic = content["metadata"].get("topic", "Unknown Topic")
            
            context_part = f"\n--- {content_type.upper()}: {topic} ---\n"
            context_part += content["text"]
            
            # Add essential metadata information
            if "concepts" in content["metadata"]:
                concepts = content["metadata"]["concepts"]
                if concepts:
                    context_part += f"\nKey Concepts: {', '.join(concepts)}"
            
            if "formulas" in content["metadata"]:
                formulas = content["metadata"]["formulas"]
                if formulas:
                    context_part += f"\nRelevant Formulas: {'; '.join(formulas[:2])}"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_messages(
        self, 
        question: str, 
        context: str, 
        conversation_history: List[Dict] = None
    ) -> List[Dict]:
        """Create message structure for GPT"""
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                messages.append(msg)
        
        # Add current question with direct academic instruction context
        user_message = f"""
Student Question: {question}

Course Material and Context:
{context}

Provide direct academic instruction:

1. **Concept Explanation**: Clearly explain relevant theory and principles with LaTeX formatting
2. **Solution Steps**: Provide step-by-step approach for problems with calculations in LaTeX
3. **Formulas**: Include relevant equations with variable definitions in proper LaTeX format

IMPORTANT: Format ALL mathematical expressions using LaTeX notation:
- Use $expression$ for inline mathematical expressions
- Use $$expression$$ for display equations
- Format numerical values, variables, units, and symbols in LaTeX
- Examples: $\sigma = \frac{{F}}{{A}}$, $E = 200 \text{{ GPa}}$, $\theta = 45°$

Focus on clear, direct instruction without guided discovery, scaffolding, or reflection questions.
"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _create_general_messages(
        self, 
        question: str, 
        conversation_history: List[Dict] = None
    ) -> List[Dict]:
        """Create message structure for general (non-course) questions"""
        
        # Strict academic system prompt for non-course questions with LaTeX formatting
        general_system_prompt = """
You are ARIA, an Academic Tutor for Statics & Mechanics of Materials. You respond ONLY to questions related to statics and mechanics of materials.

For ANY question not related to statics and mechanics of materials, respond exactly:
"This question is not relevant to the course."

Do not provide assistance, explanations, or guidance for topics outside of statics and mechanics of materials. If the question contains any mathematical expressions, format them using LaTeX notation with $expression$ for inline math and $$expression$$ for display equations.
"""
        
        messages = [
            {"role": "system", "content": general_system_prompt}
        ]
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                messages.append(msg)
        
        # For non-course questions, use strict response
        messages.append({"role": "user", "content": question})
        
        return messages
    
    def _deduplicate_content(self, content_list: List[Dict]) -> List[Dict]:
        """Remove duplicate content based on text similarity"""
        unique_content = []
        seen_texts = set()
        
        for content in content_list:
            # Create a simple hash of the text for deduplication
            text_hash = hash(content["text"][:300])  # Increased for better deduplication
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_content.append(content)
        
        return unique_content
    
    def _validate_source_document(self, content: Dict, source_name: str) -> bool:
        """Validate if a source document is correct and relevant for statics and mechanics"""
        
        try:
            if not isinstance(content, dict) or not source_name:
                return False
            
            metadata = content.get("metadata", {})
            content_text = content.get("text", "")
            
            # Define valid course topics and keywords
            valid_topics = [
                "statics", "mechanics", "materials", "stress", "strain", "beam", "truss",
                "equilibrium", "force", "moment", "torsion", "axial", "flexural", "shear",
                "deformation", "deflection", "loading", "support", "reaction", "analysis",
                "structural", "engineering", "mechanics of materials", "strength of materials"
            ]
            
            # Check if source name contains valid course-related terms
            source_lower = source_name.lower()
            has_valid_topic = any(topic in source_lower for topic in valid_topics)
            
            # Check metadata for course relevance
            topic = metadata.get("topic", "").lower()
            content_type = metadata.get("content_type", "").lower()
            
            # Validate based on topic and content type
            if topic and any(valid_topic in topic for valid_topic in valid_topics):
                has_valid_topic = True
            
            # Check content text for course-related keywords (sample check)
            if content_text and len(content_text) > 50:
                text_lower = content_text[:500].lower()  # Check first 500 chars
                text_has_valid_content = any(topic in text_lower for topic in valid_topics[:10])  # Check key topics
                if text_has_valid_content:
                    has_valid_topic = True
            
            # Exclude obviously incorrect sources
            invalid_indicators = [
                "cooking", "recipe", "photography", "music", "art", "biology", "chemistry",
                "weather", "climate", "software", "programming", "fitness", "health",
                "marketing", "business", "finance", "literature", "history", "geography"
            ]
            
            has_invalid_content = any(indicator in source_lower for indicator in invalid_indicators)
            
            # Final validation decision
            is_valid = has_valid_topic and not has_invalid_content
            
            logger.debug(f"Source validation for '{source_name}': valid_topic={has_valid_topic}, invalid_content={has_invalid_content}, result={is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Error validating source '{source_name}': {e}")
            return False
    
    def _format_detailed_source_reference(self, content: Dict, clean_source: str) -> str:
        """Create detailed source reference with PDF document and section information"""
        
        try:
            metadata = content.get("metadata", {})
            
            # Get additional metadata for detailed attribution
            page_number = metadata.get("page", metadata.get("page_number", ""))
            section = metadata.get("section", metadata.get("chapter", ""))
            topic = metadata.get("topic", "")
            
            # Build detailed reference
            reference_parts = []
            
            # Start with the main source document
            if clean_source:
                reference_parts.append(clean_source)
            
            # Add section information if available
            if section and isinstance(section, str) and section.strip():
                section_clean = section.strip()
                if not any(skip in section_clean.lower() for skip in ['unknown', 'none', 'n/a']):
                    reference_parts.append(f"Section: {section_clean}")
            
            # Add page information if available
            if page_number and str(page_number).strip():
                page_str = str(page_number).strip()
                if page_str.isdigit() or any(char.isdigit() for char in page_str):
                    reference_parts.append(f"p. {page_str}")
            
            # Add topic for context if it's different from source name
            if topic and isinstance(topic, str) and topic.strip():
                topic_clean = topic.strip()
                if topic_clean.lower() not in clean_source.lower() and len(topic_clean) > 3:
                    reference_parts.append(f"({topic_clean})")
            
            # Combine parts with improved formatting
            if len(reference_parts) > 1:
                detailed_ref = reference_parts[0]
                
                # Separate topic (in parentheses) from other parts
                topic_parts = [part for part in reference_parts[1:] if part.startswith("(")]
                other_parts = [part for part in reference_parts[1:] if not part.startswith("(")]
                
                # Add non-topic parts first
                if other_parts:
                    detailed_ref += " - " + ", ".join(other_parts)
                
                # Add topic parts at the end
                if topic_parts:
                    if other_parts:
                        detailed_ref += " " + " ".join(topic_parts)
                    else:
                        detailed_ref += " - " + " ".join(topic_parts)
                
                return detailed_ref
            else:
                return reference_parts[0] if reference_parts else clean_source
                
        except Exception as e:
            logger.warning(f"Error creating detailed source reference: {e}")
            return clean_source

    def _format_source_references(self, content_list: List[Dict]) -> str:
        """Format source references with enhanced PDF document attribution and section details"""
        
        try:
            # Input validation
            if not content_list or not isinstance(content_list, list):
                logger.warning(f"Invalid content_list for source formatting: {type(content_list)}")
                return ""
            
            detailed_sources = []
            valid_sources_found = 0
            
            # Debug logging
            logger.info(f"Formatting detailed sources for {len(content_list)} content items")
            
            for i, content in enumerate(content_list):
                try:
                    if not isinstance(content, dict):
                        logger.warning(f"Skipping non-dict content item {i}: {type(content)}")
                        continue
                        
                    metadata = content.get("metadata", {})
                    if not isinstance(metadata, dict):
                        logger.warning(f"Content {i} has invalid metadata: {type(metadata)}")
                        continue
                        
                    logger.debug(f"Content {i} metadata keys: {list(metadata.keys())}")
                    
                    # Try multiple possible source field names with validation
                    source_file = None
                    for field_name in ["source_file", "source", "filename", "file_name"]:
                        field_value = metadata.get(field_name) or content.get(field_name, "")
                        if field_value and isinstance(field_value, str) and field_value.strip():
                            source_file = field_value.strip()
                            break
                    
                    logger.debug(f"Content {i} source_file: {source_file}")
                    
                    if source_file:
                        # Extract just the filename from the full path
                        import os
                        filename = os.path.basename(source_file)
                        
                        # Validate filename
                        if not filename or len(filename) < 1:
                            logger.warning(f"Invalid filename extracted from {source_file}")
                            continue
                        
                        # Clean up the source file name for better readability
                        clean_source = filename.replace(".json", "").replace(".pdf", "").replace("_", " ").replace("-", " ")
                        
                        # Convert to title case and clean up specific patterns
                        clean_source = clean_source.title().strip()
                        
                        # Validate clean_source
                        if not clean_source or len(clean_source) < 2:
                            logger.warning(f"Source name too short after cleaning: '{clean_source}'")
                            continue
                        
                        # Handle specific course material patterns
                        if "Beam Bending Members" in clean_source:
                            clean_source = "Beams: Flexural Stresses and Strains"
                        elif "Axial Force Members" in clean_source:
                            clean_source = "Axial Loading: Stress and Strain"
                        elif "Torsion Members" in clean_source:
                            clean_source = "Torsion: Shear Stress and Angle of Twist"
                        elif "Extracted" in clean_source:
                            # Remove "Extracted" suffix and clean up
                            clean_source = clean_source.replace(" Extracted", "").strip()
                        
                        # Validate source document before adding
                        if clean_source and len(clean_source) >= 2 and self._validate_source_document(content, clean_source):
                            # Create detailed source reference
                            detailed_ref = self._format_detailed_source_reference(content, clean_source)
                            detailed_sources.append(detailed_ref)
                            valid_sources_found += 1
                            logger.debug(f"Added valid detailed source: {detailed_ref}")
                        else:
                            logger.debug(f"Rejected invalid source: {clean_source}")
                            
                    else:
                        # Fallback: try to extract from topic or content_type
                        topic = metadata.get("topic", "")
                        content_type = metadata.get("content_type", "")
                        
                        if topic and isinstance(topic, str) and topic.strip():
                            content_type_str = content_type.title() if isinstance(content_type, str) else "Content"
                            fallback_source = f"{content_type_str}: {topic.strip()}"
                            
                            # Validate fallback source
                            if self._validate_source_document(content, fallback_source):
                                # Create detailed reference for fallback too
                                detailed_ref = self._format_detailed_source_reference(content, fallback_source)
                                detailed_sources.append(detailed_ref)
                                valid_sources_found += 1
                                logger.debug(f"Added valid detailed fallback source: {detailed_ref}")
                            else:
                                logger.debug(f"Rejected invalid fallback source: {fallback_source}")
                            
                except Exception as item_error:
                    logger.warning(f"Error processing content item {i}: {item_error}")
                    continue
            
            # Generate final result with detailed attribution
            if detailed_sources and valid_sources_found > 0:
                # Remove duplicates while preserving order and detail
                unique_sources = []
                seen_base_sources = set()
                
                for detailed_source in detailed_sources:
                    # Extract base source name for deduplication
                    base_source = detailed_source.split(' - ')[0].split('(')[0].strip()
                    if base_source not in seen_base_sources:
                        seen_base_sources.add(base_source)
                        unique_sources.append(detailed_source)
                
                # Format with enhanced presentation
                if len(unique_sources) == 1:
                    result = unique_sources[0]
                else:
                    # Use numbered list for multiple sources
                    numbered_sources = [f"[{i+1}] {source}" for i, source in enumerate(unique_sources)]
                    result = "; ".join(numbered_sources)
                
                # Final validation of result
                if result and len(result) > 0:
                    logger.info(f"Successfully formatted {valid_sources_found} detailed sources with attribution")
                    return result
                else:
                    logger.warning("Source formatting produced empty result after validation")
                    return ""
            else:
                logger.info("No valid sources found after validation")
                return ""
                
        except Exception as e:
            logger.error(f"Error in _format_source_references: {e}")
            return ""
    
    def _extract_concepts_from_content(self, content_list: List[Dict]) -> List[str]:
        """Extract unique concepts from retrieved content"""
        concepts = set()
        for content in content_list:
            if "concepts" in content["metadata"]:
                concepts.update(content["metadata"]["concepts"])
        return list(concepts)
    
    def _is_question_course_relevant(self, question: str) -> bool:
        """Check if the question is relevant to statics and mechanics course materials with enhanced validation"""
        
        try:
            # Input validation
            if not question or not isinstance(question, str):
                logger.warning(f"Invalid question input: {type(question)} - {question}")
                return False
            
            # Clean and validate question
            question_clean = question.strip()
            if len(question_clean) < 3:  # Too short to be meaningful
                logger.info("Question too short to determine relevance")
                return False
            
            # Convert question to lowercase for case-insensitive matching
            question_lower = question_clean.lower()
            
            # Define course-relevant keywords with enhanced coverage
            statics_keywords = [
                'static', 'statics', 'equilibrium', 'force', 'forces', 'moment', 'moments',
                'torque', 'reaction', 'reactions', 'support', 'supports', 'free body diagram',
                'fbd', 'beam', 'beams', 'truss', 'trusses', 'frame', 'frames', 'joint',
                'joints', 'pin', 'roller', 'fixed', 'cantilever', 'distributed load',
                'point load', 'resultant', 'equilibrium equations'
            ]
            
            mechanics_keywords = [
                'stress', 'strain', 'deformation', 'deflection', 'bending', 'shear',
                'tension', 'compression', 'torsion', 'axial', 'flexural', 'elastic',
                'modulus', 'material', 'materials', 'mechanics', 'strength', 'loading',
                'load', 'loads', 'pressure', 'normal', 'tangential', 'principal',
                'mohr', 'circle', 'yield', 'failure', 'safety', 'factor', 'hooke',
                'poisson', 'bulk modulus', 'shear modulus'
            ]
            
            engineering_keywords = [
                'engineering', 'structural', 'mechanical', 'civil', 'design',
                'analysis', 'calculate', 'calculation', 'solve', 'problem',
                'diagram', 'section', 'cross-section', 'area', 'inertia',
                'centroid', 'geometry', 'coordinate', 'axis', 'axes', 'member',
                'element', 'structure', 'displacement', 'rotation'
            ]
            
            # Combine all keywords
            all_keywords = statics_keywords + mechanics_keywords + engineering_keywords
            
            # Enhanced keyword matching with word boundaries and context validation
            import re
            keyword_found = False
            matched_keywords = []
            
            for keyword in all_keywords:
                # Use word boundaries to avoid partial matches in unrelated words
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, question_lower):
                    matched_keywords.append(keyword)
                    keyword_found = True
            
            # Additional context validation for potentially ambiguous keywords
            if keyword_found:
                # Check for context that suggests non-engineering usage
                non_engineering_contexts = [
                    'photography', 'photo', 'camera', 'sunbeam', 'light beam',
                    'gym', 'workout', 'exercise', 'fitness', 'health', 'training',
                    'cooking', 'recipe', 'food', 'kitchen',
                    'weather', 'temperature', 'climate',
                    'programming', 'software', 'code', 'computer',
                    'music', 'art', 'biology', 'chemistry', 'physics' # except when combined with engineering
                ]
                
                for context in non_engineering_contexts:
                    if context in question_lower:
                        logger.debug(f"Found non-engineering context '{context}' despite keyword matches: {matched_keywords}")
                        # Only reject if the engineering keywords are likely coincidental
                        coincidental_keywords = ['beam', 'force', 'load', 'stress', 'pressure', 'calculate']
                        if any(kw in coincidental_keywords for kw in matched_keywords):
                            logger.debug("Keywords appear to be coincidental in non-engineering context")
                            return False
                
                logger.debug(f"Found course keywords: {matched_keywords}")
                return True
            
            # Additional check for common engineering units and symbols
            engineering_units = [
                'kn', 'kip', 'lb', 'lbs', 'newton', 'newtons', 'n', 'pa', 'psi',
                'mpa', 'gpa', 'kpa', 'mm', 'cm', 'm', 'ft', 'in', 'inch', 'inches',
                'degree', 'degrees', 'radian', 'radians', 'kg', 'gram', 'ton',
                'mm^2', 'cm^2', 'm^2', 'in^2', 'ft^2'  # Added area units
            ]
            
            for unit in engineering_units:
                pattern = r'\b' + re.escape(unit) + r'\b'
                if re.search(pattern, question_lower):
                    logger.debug(f"Found engineering unit: {unit}")
                    return True
            
            # Check for mathematical expressions common in engineering
            math_patterns = ['σ', 'τ', 'ε', 'δ', 'θ', 'φ', 'ω', 'α', 'β', 'γ', 'Δ', 'ρ']
            for pattern in math_patterns:
                if pattern in question:
                    logger.debug(f"Found engineering symbol: {pattern}")
                    return True
            
            # Check for common engineering formulas or expressions
            formula_patterns = [
                r'f\s*=\s*ma', r'σ\s*=\s*f/a', r'e\s*=\s*\d+\s*gpa',
                r'\d+\s*kn', r'\d+\s*mpa', r'\d+\s*psi', r'\d+\s*lb',
                r'moment.*about', r'sum.*forces', r'equilibrium.*equation'
            ]
            
            for pattern in formula_patterns:
                if re.search(pattern, question_lower):
                    logger.debug(f"Found engineering formula pattern: {pattern}")
                    return True
            
            logger.debug("No course-relevant indicators found in question")
            return False
            
        except Exception as e:
            logger.error(f"Error in course relevance detection: {e}")
            # Default to False for safety - no sources for uncertain cases
            return False
    
    def _validate_and_check_course_relevance(self, question: str) -> bool:
        """Wrapper method with additional validation for course relevance checking"""
        
        try:
            # Pre-validation checks
            if not question:
                logger.warning("Empty question provided for relevance check")
                return False
                
            if not isinstance(question, str):
                logger.error(f"Question must be string, got {type(question)}: {question}")
                return False
                
            question_stripped = question.strip()
            if not question_stripped:
                logger.warning("Question contains only whitespace")
                return False
            
            # Perform the actual relevance check
            is_relevant = self._is_question_course_relevant(question_stripped)
            
            # Log the decision for audit purposes
            logger.info(f"Course relevance decision: {is_relevant} for question: '{question_stripped[:100]}{'...' if len(question_stripped) > 100 else ''}'")
            
            return is_relevant
            
        except Exception as e:
            logger.error(f"Critical error in course relevance validation: {e}")
            # Default to False to prevent sources being added incorrectly
            return False
    
    def _suggest_review_materials(self, content_list: List[Dict]) -> List[str]:
        """Suggest review materials based on retrieved content"""
        topics = set()
        for content in content_list:
            topic = content["metadata"].get("topic", "")
            if topic:
                topics.add(topic)
        return list(topics)
    

    
    def _log_interaction(self, question: str, response: str, context: List[Dict], response_time: float = None):
        """Log interactions to Supabase database for analytics and improvement"""
        try:
            context_sources = [c["metadata"].get("source_file", "unknown") for c in context]
            concepts_used = self._extract_concepts_from_content(context)
            
            # Store conversation in Supabase
            success = conversation_storage.store_conversation(
                session_id=self.session_id,
                user_question=question,
                ta_response=response,
                context_sources=context_sources,
                concepts_used=concepts_used,
                response_time=response_time
            )
            
            if success:
                logger.info("Conversation logged to Supabase successfully")
            else:
                logger.warning("Failed to log conversation to Supabase, falling back to local logging")
                self._fallback_log_interaction(question, response, context, response_time)
                
        except Exception as e:
            logger.error(f"Error logging to Supabase: {e}")
            # Fallback to local JSON logging if Supabase fails
            self._fallback_log_interaction(question, response, context, response_time)
    
    def _fallback_log_interaction(self, question: str, response: str, context: List[Dict], response_time: float = None):
        """Fallback logging method using JSON files"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "question": question,
                "response_length": len(response),
                "context_sources": [c["metadata"].get("source_file", "unknown") for c in context],
                "concepts_used": self._extract_concepts_from_content(context),
                "response_time": response_time
            }
            
            log_file = self.base_path / "logs" / "ta_interactions.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
                
            logger.info("Conversation logged to local file successfully")
            
        except Exception as e:
            logger.error(f"Failed to log conversation even to local file: {e}")
    
    def get_session_id(self) -> str:
        """Get the current session ID"""
        return self.session_id
    
    def new_session(self) -> str:
        """Start a new session and return the new session ID"""
        self.session_id = str(uuid.uuid4())
        logger.info(f"Started new session: {self.session_id}")
        return self.session_id
