"""Tests for the v0.5 changes: no made-up answers, a small knowledge base, short stories.

1. A topic word ("stars", "tell me about the moon") used to start a poem about it.  Poems and
   stories now start only when asked for, or by a title of two or more words said in full.
2. Things Buddy knows a little about (content/talk.json "topics") get a short, true fact.
3. Questions Buddy can't answer get the honest curiosity reply; things it can't do get
   "I can't do that yet"; plain sentences ("i have a dog") are not "my ears missed that".
4. Every story is short.
5. Every new reply is in the speech inventory, so it is pre-rendered like everything else.
"""
import random
import unittest

from buddy.brain import Brain
from buddy.config import Config
from buddy.learning.progress import Progress


def make(age=5, seed=0):
    cfg = Config()
    cfg.data["child"]["age"] = age
    return Brain(cfg, rng=random.Random(seed), progress=Progress(None))


def said(r):
    return " ".join(r.texts())


class TestTopicWordsDoNotStartPoems(unittest.TestCase):
    TOPICS = ["stars", "star", "tell me about stars", "tell me about the stars", "i like stars", "i see a star",
              "the moon", "tell me about the moon", "sun", "tell me about the sun", "rain", "train", "clouds",
              "flowers", "butterfly", "food", "friends", "kite", "mango", "tell me about mango",
              "tell me about butterflies", "tell me about flowers"]

    def test_a_topic_is_answered_with_a_fact_not_a_poem(self):
        wrong = []
        for age in (3, 5, 8):
            b = make(age)
            for text in self.TOPICS:
                r = b.command(text)
                if r.intent != "topic":
                    wrong.append((age, text, r.intent))
        self.assertEqual(wrong, [])

    def test_stars_gets_the_star_fact(self):
        self.assertIn("giant balls of hot, glowing gas", said(make().command("tell me about stars")))

    def test_poems_and_stories_still_start_when_asked(self):
        cases = [("tell me a poem about stars", "poem", "Counting Stars"),
                 ("say the stars poem", "poem", "Counting Stars"),
                 ("a poem about the moon", "poem", "Hello Moon"),
                 ("hello moon", "poem", "Hello Moon"),
                 ("tell me a story about stars", "story", "Little Star"),
                 ("the thirsty crow", "story", "Thirsty Crow"),
                 ("thirsty crow", "story", "Thirsty Crow"),
                 ("tortoise and the hare", "story", "Tortoise"),
                 ("tell me a story", "story", ""),
                 ("tell me a poem", "poem", "")]
        for text, intent, title in cases:
            r = make().command(text)
            self.assertEqual(r.intent, intent, text)
            self.assertIn(title, said(r), text)

    def test_a_song_request_picks_a_poem_on_the_same_topic(self):
        self.assertIn("Counting Stars", said(make().command("sing twinkle twinkle little star")))

    def test_one_word_animal_still_makes_its_sound(self):
        self.assertEqual(make().command("crow").intent, "animal_sound")
        self.assertEqual(make().command("tiger").intent, "animal_sound")


class TestKnowledge(unittest.TestCase):
    def test_tell_me_about_known_things(self):
        cases = {"tell me about dinosaurs": "dinosaurs", "what is a volcano": "volcano",
                 "tell me about space": "space", "tell me about planets": "planets",
                 "what is gravity": "gravity", "tell me about the ocean": "ocean", "what is a tree": "trees",
                 "what is a heart": "heart", "tell me about the heart": "heart", "what is a fish": "fish",
                 "what is the moon": "moon", "what is the earth": "earth", "tell me about trains": "train",
                 "what is a computer": "computer", "tell me about rainbows": "rainbow", "what is a star": "stars"}
        for text, tid in cases.items():
            r = make().command(text)
            self.assertEqual((r.intent, r.meta.get("id")), ("topic", tid), text)

    def test_star_and_heart_mean_the_shape_only_when_shape_is_said(self):
        self.assertIn("five points", said(make().command("what is a star shape")))
        self.assertIn("means love", said(make().command("tell me about the heart shape")))
        self.assertIn("four sides", said(make().command("what is a square")))  # unambiguous shapes unchanged

    def test_animals_answer_what_is_a(self):
        r = make().command("what is a tiger")
        self.assertEqual((r.intent, r.meta.get("animal")), ("animal_fact", "tiger"))

    def test_why_and_how_questions_are_not_answered_with_a_general_fact(self):
        # a fact about the moon does not answer "why is the moon round"
        for text in ["why is the moon round", "how do trains work", "where do stars go",
                     "how far is the moon", "why is the moon made of cheese"]:
            r = make().command(text)
            self.assertIn(r.intent, ("curiosity", "faq"), text)
            self.assertNotIn("Isn't that amazing", said(r) if r.intent == "curiosity" else "", text)
        self.assertEqual(make().command("why is the moon made of cheese").intent, "curiosity")

    def test_new_faq_answers(self):
        cases = {"how many legs does a spider have": "eight legs", "what is the biggest planet": "Jupiter",
                 "what colour is the sun": "yellow or white", "what colour is the moon": "grey",
                 "what shape is the moon": "round", "why do dogs bark": "Dogs bark to talk"}
        for text, words in cases.items():
            r = make().command(text)
            self.assertEqual(r.intent, "faq", text)
            self.assertIn(words, said(r), text)

    def test_questions_about_buddy_get_plain_answers(self):
        for text in ["what is your favourite colour", "are you a robot", "where do you live", "who made you",
                     "what is my name"]:
            r = make().command(text)
            self.assertEqual(r.intent, "faq", text)
            self.assertNotIn("Isn't that amazing", said(r), text)

    def test_existing_lessons_still_win(self):
        self.assertEqual(make().command("what colour is the sky").intent, "color_of")
        self.assertEqual(make().command("what is your name").intent, "name")
        self.assertEqual(make().command("teach me colours").intent, "colors")
        self.assertEqual(make().command("tell me about lions").intent, "animal_fact")
        self.assertEqual(make().command("where is my nose").intent, "body")

    def test_another_one_after_a_topic(self):
        b = make()
        b.command("tell me about the moon")
        r = b.command("another one")
        self.assertEqual(r.intent, "topic")
        self.assertTrue(r.texts())


class TestHonestReplies(unittest.TestCase):
    def test_unknown_topics_get_curiosity_not_my_ears_missed_that(self):
        for text in ["tell me about pirates", "tell me about florp", "what is the capital of india",
                     "who is the president", "what time is it", "what is the weather today"]:
            self.assertEqual(make().command(text).intent, "curiosity", text)

    def test_things_buddy_cannot_do(self):
        for text in ["spell cat", "call mummy", "turn on the light", "play hide and seek", "draw a cat",
                     "play a cartoon"]:
            r = make().command(text)
            self.assertEqual(r.intent, "cant_do", text)
            self.assertIn(said(r), make().content.phrases["cant_do"], text)

    def test_plain_sentences_get_a_friendly_reply(self):
        for text in ["i have a dog", "my name is riya", "i am five years old", "we went to the park",
                     "my dog is called max", "i went to school today"]:
            self.assertEqual(make().command(text).intent, "chat", text)

    def test_being_hurt_goes_to_a_grown_up(self):
        for text in ["i fell down and hurt my knee", "i am sick", "my tummy hurts"]:
            r = make().command(text)
            self.assertIn("tell a grown-up", said(r), text)

    def test_gibberish_still_gets_the_fallback(self):
        fallback = set(make().content.phrases["not_understood"] + make().content.phrases["not_understood_help"])
        for text in ["florp bimble wug", "asdfgh", "my florp bimble", "i like florp", "eeee", "sta"]:
            r = make().command(text)
            self.assertIn(said(r), fallback, text)

    def test_plain_sentences_are_background_talk_in_the_follow_up_window(self):
        b = make()
        b.command("tell me a joke")
        self.assertIsNone(b.follow_up("we went to the park"))
        self.assertIsNone(b.follow_up("my dog is called max"))

    def test_a_plain_sentence_does_not_wake_buddy(self):
        b = make()
        woke, _rest, _score = b.wake.split("buddy the dog is asleep")
        self.assertFalse(woke)


class TestShortStories(unittest.TestCase):
    def test_every_story_is_short(self):
        for s in make().content.stories:
            words = sum(len(p.split()) for p in s["paragraphs"])
            self.assertLessEqual(len(s["paragraphs"]), 3, s["id"])
            self.assertLessEqual(words, 90, s["id"])
            self.assertIn("The end.", s["paragraphs"][-1], s["id"])


class TestInventory(unittest.TestCase):
    """Buddy only says pre-rendered sentences: every new reply must be in the inventory."""

    INPUTS = ["tell me about stars", "what is a volcano", "what is your favourite colour", "spell cat",
              "i have a dog", "i am sick", "tell me about pirates", "what shape is the moon",
              "how many legs does a spider have", "tell me a story", "the thirsty crow", "what is a tiger"]

    def test_replies_are_in_the_speech_inventory(self):
        b = make()
        inventory = b.skills.all_phrases()
        for text in self.INPUTS:
            for line in make().command(text).texts():
                self.assertIn(line, inventory, (text, line))

    def test_every_topic_and_story_line_is_in_the_inventory(self):
        b = make()
        inventory = b.skills.all_phrases()
        for tp in b.content.talk["topics"]:
            self.assertIn(tp["say"], inventory)
        for s in b.content.stories:
            for p in s["paragraphs"]:
                self.assertIn(p, inventory)
        for key in ("cant_do", "chat_ack", "feel_hurt"):
            for line in b.content.phrases[key]:
                self.assertIn(line, inventory)


if __name__ == "__main__":
    unittest.main()
