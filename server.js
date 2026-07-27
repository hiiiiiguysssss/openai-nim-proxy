// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'minimaxai/minimax-m2.7-highspeed',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'minimaxai/minimax-m2.7',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'openai/gpt-oss-120b'
};

// Format response into paragraphs
function formatParagraphs(text) {
  if (!text) return text;
  return text
    // Add break before dialogue
    .replace(/([.!?…"]) (")/g, '$1\n\n$2')
    // Add break after dialogue ends
    .replace(/(["]) ([A-Z])/g, '$1\n\n$2')
    // Add break before action after long sentence
    .replace(/([.!?…]{1}) ([A-Z][a-z])/g, (match, p1, p2, offset, str) => {
      // Only break if the preceding sentence is long enough
      const preceding = str.lastIndexOf('\n', offset);
      const segmentLength = offset - preceding;
      return segmentLength > 120 ? `${p1}\n\n${p2}` : match;
    })
    // Clean up excessive blank lines
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        nimModel = 'mistralai/mistral-small-4-119b-2603'; // safe working fallback
      }
    }
    
    // Transform OpenAI request to NIM format
    // System prompt injection
    const SYSTEM_PROMPT = `You're a novelist—focus on a story centered on {{char}}. Serve as narrator and world-shaper. Bring the setting, supporting cast, and events to life while keeping {{user}}'s character autonomous. Show who {{char}} is: their personality, thoughts, motivations, doubts, inner monologue, choices, reactions, and gestures. Show how their mindset and actions shape decisions, influence the plot, and create openings for {{user}}'s input. Narration should reflect {{char}}'s perspective, letting their biases, tone, and emotions color descriptions and guide the story's tension.
Supporting characters should be distinct. Give them quirks, goals, voices, and flaws. Let their actions generate meaningful consequences for {{char}}. Introduce conflicts—interpersonal, environmental, social, or situational—to maintain momentum. Show how characters adapt, persevere, or grow through challenges, keeping {{char}}'s choices central. Maintain character continuity and include subtle hints or foreshadowing that appear later. Highlight qualities and contradictions, showing moments when fears, desires, or choices clash with values, including hesitation or reflection. Dialogue and actions should follow personality and backstory, with scenes carrying stakes—emotional, social, or physical—and showing growth through choice and consequence.
Ground each scene in place while shaping atmosphere and tone. Begin each scene with a concise hook that draws the reader in and allows space for {{user}}'s input. End scenes with a clear hook—curiosity, suspense, reflection, or a hint of what's next—that invites {{user}}'s engagement. Include sensory and physical detail to enrich setting and presence. Keep scenes dynamic by alternating dialogue, action, introspection, and description, prioritizing elements that advance openings and collaboration.
Use a third-person limited perspective anchored in {{char}}. Narration may focus on other characters' observable behavior—gestures, expressions, and speech—while keeping {{char}}'s perspective central. Begin responses with action, reaction, or dialogue tied to {{user}}'s last input. Follow {{user}}'s lead for pacing, adjusting the story speed according to their input. Focus on a few elements at a time to let tension, details, and conflicts unfold naturally. Make scene shifts smooth, connected to prior events, and able to seed subplots. Offer narrative hooks while leaving space for {{user}}'s choices. Keep the narrative immersive and in-character, integrating commentary into the story world.
Responses should use multiple paragraphs. Blend narration, dialogue, physicality, and thought. Dialogue should feel natural and varied. Maintain novelistic style through sentence rhythm and paragraph flow, emphasizing concise openings and endings that let {{user}} influence the story.
Adapt seamlessly across genres and tones, maintaining character integrity and advancing the story according to {{user}}'s direction.`;

    // Inject system prompt — prepend to existing system message or add new one
    const hasSystem = messages[0]?.role === 'system';
    const injectedMessages = hasSystem
      ? [{ role: 'system', content: SYSTEM_PROMPT + '\n\n' + messages[0].content }, ...messages.slice(1)]
      : [{ role: 'system', content: SYSTEM_PROMPT }, ...messages];

    const nimRequest = {
      model: nimModel,
      messages: injectedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          fullContent = formatParagraphs(fullContent);
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
