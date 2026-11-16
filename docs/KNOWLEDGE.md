# v0.5: no made-up answers, a small knowledge base, short stories (16 Nov 2026)

## What was wrong
- **A topic word started a poem.** "tell me about stars", "the moon", "rain", "train", "mango"
  and similar inputs all recited a poem. Every poem had one-word aliases ("stars", "moon"), and
  `Matcher._decide` started any title it matched, even when nobody asked for a poem. Stray sounds
  did the same at ages 3–7: "eeee" and "sta" started poems, and "brilliant", "flibber" and
  "traceback" started stories.
- **Wrong activity for a known word.** "what is a star" and "tell me about the heart" gave the
  *shape* lesson. "what colour is the sun" recited every colour, "what shape is the moon"
  recited every shape, "spell cat" made a cat sound, "my dog is called max" said "My pleasure"
  ("max" was snapped to "thanks"), "thirsty crow" made a crow sound, and "why is the moon made
  of cheese" told a joke ("cheese" was guessed as "jokes").
- **"Oops, my ears missed that" for things Buddy heard fine.** This happened for "tell me about
  dinosaurs" and for plain sentences such as "i have a dog".
- **Long stories:** 107–167 words in 4–6 paragraphs.

## What changed
| Where | Change |
|---|---|
| `content/talk.json` → `topics` | **New.** 50 short, true facts a child asks about: sky and space, weather, nature, small creatures that aren't in `animals.json`, the body, transport, school, family, and so on. |
| `content/talk.json` → `faq` | 19 new questions (spider legs, biggest planet, colour or shape of the sun and moon, "are you a robot", "what is your favourite colour", "what is my name", …). Entries marked `"plain": true` are not followed by "Isn't that amazing?". |
| `content/phrases.json` | New `cant_do` ("I can't do that yet…"), `chat_ack` (for a plain sentence) and `feel_hurt` ("please tell a grown-up"). |
| `content/stories.json` | All 11 stories were shortened to 3 paragraphs and 62–85 words (about half). The ids, titles, aliases and morals are unchanged. |
| `nlu/matcher.py` | See the rules below. |
| `skills/handlers.py` | New `do_topic`, `do_cant_do` and `do_chat`. Topic facts are in `all_phrases()`, so `build-content` pre-renders them. |
| `brain.py` | "chat" is treated like "unknown" by the wake check and the follow-up window, so background talk is still ignored. |

### Matching rules (nlu/matcher.py)
1. **A poem or story starts only when asked for** (story/poem/rhyme/song), **or when a title of
   two or more different words is said in full** ("thirsty crow", "hello moon", "tortoise and the
   hare"). A single word never starts one.
2. **Topics:** these phrasings get the topic's fact: "tell me about X", "what is/are (a/the) X",
   or X on its own ("stars", "the moon", "i like trains").
   - A word Buddy only guessed at never counts.
   - A "why / how / where / when" question never gets a topic fact, because a fact about the moon
     does not answer "why is the moon round". Those questions go to the FAQ, or to the honest
     curiosity reply.
   - Topic words are protected from snapping but are **not** snapping targets. As targets, they
     pulled a child's "monty" (monkey) to "mountain".
3. "tell me about X" for an unknown X → the curiosity reply, not "my ears missed that".
4. Requests Buddy can't do (spell, draw, call mummy, turn on the light, hide and seek, cartoons,
   order/buy) → `cant_do`. Questions such as weather or time keep the curiosity reply.
5. A plain sentence of real English ("i have a dog", "my name is riya") → `chat_ack`. One unknown
   word is allowed in a sentence of 4+ words (a name). Gibberish still gets the fallback.
6. Hurt or sick ("i hurt my knee", "my tummy hurts") → tell a grown-up.
7. star, heart and diamond mean the shape only when "shape" is said. "What colour/shape is X"
   for an X Buddy doesn't know no longer recites the whole lesson.
8. A one-key-word FAQ question ("who made you") matches only word for word. A "why" question
   never starts an activity on a guessed keyword.

## Adding a topic
Add one line to `topics` in `content/talk.json`:
`{"id": "owl", "names": ["owl", "owls"], "say": "…"}`. Don't reuse an animal from `animals.json`,
because animals already have facts. Then re-run `python -m buddy build-content` to render the new
sentence.

## Measured
- **Unit tests:** 134 → 157, all passing (+23 in `tests/test_knowledge.py`).
- **Original tests:** no assertion changed. One `@expectedFailure` was removed from
  `test_known_residual_a_vocabulary_word_can_still_carry_a_question`, because the case it
  recorded ("why is the moon made of cheese" → joke) is now fixed. Python counts an unexpected
  success as a failure.
- **Sweep of every phrase in the original tests, ages 3/5/8, old vs new:** "monty sound" →
  monkey is unchanged, and so is every child-pronunciation case. Changes:
  - random words no longer start stories or poems;
  - one-word topics give facts;
  - plain sentences get `chat`;
  - "call mummy / order a pizza / turn on the tv" → `cant_do`;
  - "why do dogs bark" → its new FAQ answer.
- **Short mumbles (ages 3–8):** "ngggh mmmph" → months and "wawawa / gah" → feeling at ages
  3–4 are unchanged. Those are the known residuals in `docs/FIXES.md`.

## Needs doing on the PC
Run `python -m buddy build-content` to render the new and shortened sentences (only new
sentences are rendered). Until then, `run` logs a warning for each missing clip.
