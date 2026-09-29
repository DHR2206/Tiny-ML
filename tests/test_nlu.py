import unittest

from buddy.content_store import Content
from buddy.nlu.matcher import Matcher
from buddy.nlu.normalize import normalize, parse_numbers
from buddy.nlu.phonetics import ChildPhonology, g2p

# (utterance after the wake phrase, expected intent, expected slots subset)
CASES = [
    ("play a poem", "poem", {}), ("say a rhyme", "poem", {}), ("mango poem", "poem", {"id": "mango"}),
    ("tell me the kite poem", "poem", {"id": "kite"}), ("sing a song", "song", {}), ("play music", "song", {}),
    ("tell me a story", "story", {}), ("story please", "story", {}), ("bedtime story", "story", {}),
    ("tell me the lion and the mouse story", "story", {"id": "lion_mouse"}),
    ("the thirsty crow story", "story", {"id": "thirsty_crow"}),
    ("tortoise and the hare story", "story", {"id": "tortoise_hare"}),
    ("play alphabet", "alphabet", {}), ("abc", "alphabet", {}), ("say the alphabets", "alphabet", {}),
    ("a b c", "alphabet", {}), ("phonics", "phonics", {}), ("letter sounds", "phonics", {}),
    ("what is b for", "phonics", {"letter": "b"}), ("what sound does letter m make", "phonics", {"letter": "m"}),
    ("count to 100", "count", {"end": 100}), ("count till hundred", "count", {"end": 100}),
    ("count to twenty", "count", {"end": 20}), ("count from 10 to 20", "count", {"start": 10, "end": 20}),
    ("count backwards from 10", "count", {"backwards": True}), ("count by twos", "count", {"step": 2}),
    ("count by tens", "count", {"step": 10}), ("let's count", "count", {}), ("numbers", "count", {}),
    ("animal sounds", "animal_sound", {}), ("play animal sounds", "animal_sound", {}),
    ("what does a cow say", "animal_sound", {"animal": "cow"}),
    ("how does the lion go", "animal_sound", {"animal": "lion"}),
    ("dog sound", "animal_sound", {"animal": "dog"}), ("make a doggy noise", "animal_sound", {"animal": "dog"}),
    ("what does the elephant say", "animal_sound", {"animal": "elephant"}), ("monkey", "animal_sound", {"animal": "monkey"}),
    ("tell me about tigers", "animal_fact", {"animal": "tiger"}),
    ("colours", "colors", {}), ("teach me colors", "colors", {}), ("what color is a banana", "color_of", {"thing": "banana"}),
    ("shapes", "shapes", {}), ("what is a triangle shape", "shapes", {"shape": "triangle"}), ("body parts", "body", {}),
    ("days of the week", "days", {}), ("months of the year", "months", {}), ("opposites", "opposites", {}),
    ("table of 5", "table", {"n": 5}), ("five times table", "table", {"n": 5}), ("tables", "table", {}),
    ("what is 3 plus 4", "math", {"a": 3, "b": 4, "op": "plus"}), ("what is two plus two", "math", {"a": 2, "b": 2}),
    ("6 times 7", "math", {"op": "times"}), ("what is ten minus three", "math", {"op": "minus"}),
    ("let's play a quiz", "quiz", {}), ("play a game", "quiz", {}), ("animal quiz", "quiz", {"domain": "animals"}),
    ("number game", "quiz", {"domain": "numbers"}), ("ask me a question", "quiz", {}),
    ("tell me a joke", "joke", {}), ("something funny", "joke", {}), ("riddle", "riddle", {}),
    ("breathe with me", "breathing", {}), ("i want to calm down", "breathing", {}), ("magic words", "manners", {}),
    ("i am sad", "feeling", {"mood": "sad"}), ("i feel angry", "feeling", {"mood": "angry"}),
    ("i'm scared", "feeling", {"mood": "scared"}), ("i'm happy", "feeling", {"mood": "happy"}),
    ("i am bored", "feeling", {"mood": "bored"}),
    ("what is your name", "name", {}), ("who are you", "name", {}), ("how old are you", "age", {}),
    ("how are you", "how_are_you", {}), ("i love you", "love", {}), ("thank you", "thanks", {}),
    ("good morning", "good_morning", {}), ("good night", "good_night", {}), ("bye bye", "goodbye", {}),
    ("hello", "hello", {}), ("sorry", "sorry", {}),
    ("stop", "stop", {}), ("be quiet", "stop", {}), ("that's enough", "stop", {}), ("louder", "volume_up", {}),
    ("volume up", "volume_up", {}), ("softer", "volume_down", {}), ("too loud", "volume_down", {}),
    ("say it again", "repeat", {}), ("repeat", "repeat", {}), ("another one", "more", {}), ("one more", "more", {}),
    ("what can you do", "help", {}), ("help", "help", {}),
    ("why is the sky blue", "faq", {"index": 0}), ("where does rain come from", "faq", {}),
    ("why do dogs bark", "curiosity", {}), ("how do airplanes fly", "curiosity", {}),
    ("where do dinosaurs live", "curiosity", {}),
]

# Typical child pronunciations (as a recogniser might spell them)
CHILD_CASES = [
    ("pway a stowy", "story"),          # gliding + cluster
    ("tell me a towy", "story"),        # cluster reduction
    ("tount to ten", "count"),          # fronting
    ("tat sound", "animal_sound"),      # fronting  cat -> tat
    ("wion sound", "animal_sound"),     # gliding  lion -> wion
    ("what does the dut say", "animal_sound"),  # fronting duck -> dut
    ("pway alfabet", "alphabet"),
    ("tell me a pome", "poem"),
    ("ewephant sound", "animal_sound"),  # gliding l -> w
    ("tell me a doke", "joke"),          # stopping/fronting j -> d
    ("wet's pway a tame", "quiz"),       # game -> tame (fronting)
    ("tolours", "colors"),               # fronting
    ("widdle", "riddle"),                # gliding r -> w
    ("monty sound", "animal_sound"),     # monkey -> monty
    ("fank you", "thanks"),              # th-fronting
    ("sapes", "shapes"),                 # depalatalisation
]

# Natural variations: the same request in other words, word orders, number forms and
# recogniser spellings ("1-10").  Handled by the shared slot extractor, not by listing phrases.
VARIATION_CASES = [
    # counting: any phrasing of a range, step or direction
    ("count from 1 to 10", "count", {"start": 1, "end": 10}), ("count 1 to 10", "count", {"start": 1, "end": 10}),
    ("can you count from 1 to 10", "count", {"start": 1, "end": 10}),
    ("start counting from 5 to 15", "count", {"start": 5, "end": 15}),
    ("count from 3 until 8", "count", {"start": 3, "end": 8}), ("count one to ten", "count", {"start": 1, "end": 10}),
    ("Count from 1-10.", "count", {"start": 1, "end": 10}), ("count 1 - 20", "count", {"start": 1, "end": 20}),
    ("count from 1 through 10", "count", {"start": 1, "end": 10}),
    ("count between 3 and 8", "count", {"start": 3, "end": 8}),
    ("count from one two ten", "count", {"start": 1, "end": 10}),  # "to" heard as "two"
    ("can't from one to ten", "count", {"start": 1, "end": 10}),  # "count" misheard: the range says it
    ("from five to fifteen", "count", {"start": 5, "end": 15}),
    ("count up to 15", "count", {"end": 15}), ("let's count to 7", "count", {"end": 7}),
    ("i want to count to twelve", "count", {"end": 12}), ("count to a hundred", "count", {"end": 100}),
    ("count by 3", "count", {"step": 3}), ("count in threes", "count", {"step": 3}),
    ("count in 5s", "count", {"step": 5}), ("count by twos to 20", "count", {"step": 2, "end": 20}),
    ("count to 20 by 2", "count", {"step": 2, "end": 20}),
    ("count down from 10", "count", {"backwards": True, "start": 10}),
    ("countdown from 5", "count", {"backwards": True, "start": 5}),
    ("count backwards from 20 to 10", "count", {"backwards": True, "start": 20, "end": 10}),
    ("count 10 to 1", "count", {"start": 10, "end": 1}),
    # sums and tables
    ("what is 2 plus 3", "math", {"a": 2, "b": 3, "op": "plus"}), ("2 + 3", "math", {"a": 2, "b": 3}),
    ("add 2 and 3", "math", {"a": 2, "b": 3, "op": "plus"}), ("how much is 5 and 2", "math", {"a": 5, "b": 2}),
    ("take away 2 from 5", "math", {"a": 5, "b": 2, "op": "minus"}), ("what is 12 plus 5", "math", {"a": 12, "b": 5}),
    ("what is 5-3", "math", {"a": 5, "b": 3, "op": "minus"}), ("3 x 4", "math", {"a": 3, "b": 4, "op": "times"}),
    ("multiply 3 by 4", "math", {"a": 3, "b": 4, "op": "times"}), ("table of 12", "table", {"n": 12}),
    ("tell me the 2 times table", "table", {"n": 2}),
    # stories, poems, letters, animals
    ("can you tell me a story", "story", {}), ("read me a story", "story", {}), ("story time", "story", {}),
    ("tell me a story about a lion", "story", {}), ("a poem about the moon", "poem", {}),  # which one: test_brain
    ("recite a poem", "poem", {}), ("can you say a rhyme", "poem", {}), ("sing me a song", "song", {}),
    ("what does the letter b say", "phonics", {"letter": "b"}), ("a for apple", "phonics", {"letter": "a"}),
    ("what sound does a cow make", "animal_sound", {"animal": "cow"}), ("how does a dog bark", "animal_sound", {"animal": "dog"}),
    ("bee sound", "animal_sound", {"animal": "bee"}), ("can you make a lion sound", "animal_sound", {"animal": "lion"}),
    ("lion facts", "animal_fact", {"animal": "lion"}), ("what noise does a pig make", "animal_sound", {"animal": "pig"}),
    # games, lessons
    ("quiz me", "quiz", {}), ("let's play", "quiz", {}), ("can we play a game", "quiz", {}),
    ("test me on colours", "quiz", {"domain": "colors"}), ("make me laugh", "joke", {}), ("give me a riddle", "riddle", {}),
    ("what colour is the sky", "color_of", {"thing": "sky"}), ("what is the opposite of hot", "opposite_of", {"word": "hot"}),
    ("tell me the months", "months", {}), ("what are the days of the week", "days", {}),
    # controls
    ("please stop", "stop", {}), ("pause", "pause", {}), ("wait", "pause", {}), ("hold on", "pause", {}),
    ("wait a minute", "pause", {}), ("continue", "continue", {}), ("go on", "continue", {}), ("keep going", "continue", {}),
    ("resume", "continue", {}), ("continue the story", "continue", {}), ("don't stop", "continue", {}),
    ("quit", "quit", {}), ("exit", "quit", {}), ("go to sleep", "quit", {}), ("i'm done", "quit", {}),
    ("repeat", "repeat", {}), ("say that again", "repeat", {}), ("what did you say", "repeat", {}),
    ("next", "more", {}), ("turn it up", "volume_up", {}), ("speak up", "volume_up", {}), ("turn it down", "volume_down", {}),
]


class TestNLU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = Matcher(Content(), age=4)
        cls.m6 = Matcher(Content(), age=6)

    def test_commands(self):
        fails = []
        for text, name, slots in CASES:
            got = self.m.parse(text)
            if got.name != name or any(got.slots.get(k) != v for k, v in slots.items()):
                fails.append((text, name, slots, got))
        acc = 1 - len(fails) / len(CASES)
        for f in fails:
            print("  FAIL", f)
        print(f"clean command accuracy: {acc:.1%} ({len(CASES)} cases)")
        self.assertGreaterEqual(acc, 0.95)

    def test_child_pronunciations(self):
        fails = [(t, n, self.m.parse(t)) for t, n in CHILD_CASES if self.m.parse(t).name != n]
        for f in fails:
            print("  CHILD FAIL", f)
        acc = 1 - len(fails) / len(CHILD_CASES)
        print(f"child-variant accuracy (age 4): {acc:.1%} ({len(CHILD_CASES)} cases)")
        self.assertGreaterEqual(acc, 0.85)

    def test_natural_variations(self):
        fails = []
        for text, name, slots in VARIATION_CASES:
            got = self.m6.parse(text)
            if got.name != name or any(got.slots.get(k) != v for k, v in slots.items()):
                fails.append((text, name, slots, got))
        for f in fails:
            print("  VARIATION FAIL", f)
        print(f"natural-variation accuracy: {1 - len(fails) / len(VARIATION_CASES):.1%} ({len(VARIATION_CASES)} cases)")
        self.assertEqual(fails, [])

    def test_repeat_of_something(self):
        got = self.m.parse("tell me the story again")
        self.assertEqual(got.name, "repeat")
        self.assertEqual(got.slots["then"].name, "story")

    def test_ordinary_words_are_not_snapped(self):
        # the first pass keeps real words; they used to become three / ten / heart / quiet / excited
        for w in ("through", "turn", "hot", "quit", "exit", "continue", "wait", "laugh", "facts", "recite"):
            toks, _ = self.m.canonicalize([w], aggressive=False)
            self.assertEqual(toks, [w])
        # ...while children's non-words still are
        self.assertEqual(self.m.canonicalize(["tount", "to", "ten"], aggressive=False)[0][0], "count")

    def test_age_schedule(self):
        young, older = ChildPhonology(3.5), ChildPhonology(8.5)
        # fronting (cat -> tat) is expected at 3.5 years, much less at 8.5
        self.assertGreater(young.similarity(g2p("tat"), g2p("cat")), older.similarity(g2p("tat"), g2p("cat")))
        # gliding (rabbit -> wabbit) is still normal at 6
        self.assertGreater(ChildPhonology(6).similarity(g2p("wabbit"), g2p("rabbit")), 0.85)

    def test_numbers(self):
        self.assertEqual([n for n, _, _ in parse_numbers(normalize("count to one hundred"))], [100])
        self.assertEqual([n for n, _, _ in parse_numbers(normalize("twenty seven and 5"))], [27, 5])
        self.assertEqual([n for n, _, _ in parse_numbers(normalize("free"), lenient=True)], [3])

    def test_answers(self):
        self.assertTrue(self.m.matches_answer("moo moo", ["moo"]))
        self.assertTrue(self.m.matches_answer("it's eight", [8]))
        self.assertTrue(self.m.matches_answer("ate", [8]))
        self.assertTrue(self.m.matches_answer("yewwow", ["yellow"]))
        self.assertFalse(self.m.matches_answer("woof", ["moo"]))
        self.assertTrue(self.m.is_dont_know("I don't know"))


if __name__ == "__main__":
    unittest.main()
