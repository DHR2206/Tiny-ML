"""Everyday English words that must NOT be "snapped" onto Buddy's vocabulary.

The matcher (nlu/matcher.py) maps unfamiliar words onto the closest word Buddy knows, using
the child-phonology distance, so that "wion" becomes "lion" and "tount" becomes "count".  With a
small vocabulary that also rewrote correct words: "through" became "three", "hot" became
"heart", "quit" became "quiet", "exit" became "excited".  So the first matching pass leaves
every word in this list alone and only snaps words that are not ordinary English.  If
nothing is understood that way, a second pass snaps every word as before.

Deliberately NOT listed: real words that are typical child versions of Buddy's keywords
("tat" for cat, "tame" for game, "top" for stop ...), so they are still snapped in the
first pass.
"""
from __future__ import annotations

_WORDS = """
a an the this that these those there here where when why how what which who whom whose
i me my mine myself you your yours yourself he him his himself she her hers herself it its itself
we us our ours ourselves they them their theirs themselves one ones some any all each every both
either neither no none nothing something anything everything someone anyone everyone nobody
somebody anybody everybody other others another such own same few many much more most less least
lot lots plenty enough several whole half
be am is are was were been being have has had having do does did done doing will would shall
should can could may might must ought need needs used
and or but nor so yet because if unless while although though since until till as than then also
too very just only even still already again ever never always often sometimes usually really
quite rather almost maybe perhaps please thanks thank yes no not okay ok yeah yep nope sure
about above across after against along among around at before behind below beneath beside
besides between beyond by down during except for from in inside into like near of off on onto
out outside over past per round through throughout to toward towards under underneath up upon
with within without away back forward forwards together apart instead else once twice

go goes went gone going come comes came coming get gets got gotten getting make makes made
making take takes took taken taking give gives gave given giving put puts putting keep keeps
kept keeping let lets letting say says said saying tell tells told telling ask asks asked
asking see sees saw seen seeing look looks looked looking watch watched watching hear hears
heard hearing listen listened listening know knows knew known knowing think thinks thought
thinking want wants wanted wanting like likes liked liking love loves loved loving need
needed feel feels felt feeling find finds found try tries tried trying use uses using help
helps helped helping show shows showed shown showing start starts started starting stop stops
stopped stopping begin begins began begun beginning finish finishes finished finishing end ends
ended ending continue continues continued continuing resume resumed pause pauses paused wait
waits waited waiting hold holds held holding hang hung carry carries carried carrying quit
quits quitting exit exits leave leaves left stay stays stayed sleep sleeps slept sleeping
wake wakes woke woken awake rest play plays played playing read reads reading write writes
wrote written writing draw draws drew drawn drawing paint painting sing sings sang sung singing
dance dances danced dancing jump jumps jumped jumping run runs ran running walk walks walked
walking sit sits sat sitting stand stands stood standing eat eats ate eaten eating drink drinks
drank drunk drinking cook cooking wash washing clean cleaning brush brushing bath bathe open
opens opened close closes closed turn turns turned turning switch switched call calls called
calling talk talks talked talking speak speaks spoke spoken speaking shout whisper cry cries
cried crying laugh laughs laughed laughing smile smiles smiled hug hugs kiss throw threw thrown
catch caught kick hit push pull build built fall falls fell fallen fly flies flew flown flying
swim swims swam swimming climb climbing ride rides rode riding drive drove buy bought bring
brought send sent learn learns learned learnt learning teach teaches taught teaching study
remember remembers forget forgets forgot forgotten understand understood mean means meant guess
count counts counted counting add adds added adding repeat repeats repeated recite recites
recited spell spelled spelling tidy pick picked choose chose chosen share shared grow grows
grew grown growing live lives lived living die move moves moved moving change changes changed
happen happens happened work works worked working fix fixed break breaks broke broken lose lost
win wins won hide hid hidden seek touch smell taste bite bit bark barks barked hop hops hopped
crawl buzz hiss roar moo quack tweet sniff blow blew breathe hear shake wave clap stamp shout
explain describe answer answers answered question questions sound sounds sounded noise noises
wonder wonders mind matter matters care cares happen skip skipped miss missed

mom mum mother dad father brother sister baby grandma grandpa granny grandmother grandfather
nana nani dadi aunt aunty uncle cousin friend friends family teacher boy boys girl girls man men
woman women kid kids child children person people name names king queen prince princess doctor
police farmer baker

house home room bed bedroom kitchen bathroom door window chair sofa floor wall roof garden yard
toy toys ball balls doll dolls teddy block blocks book books picture pictures paper pencil pen
crayon crayons bag box boxes cup cups plate spoon fork bowl bottle phone television computer
tablet light lights lamp clock towel soap toothbrush shoes shoe socks shirt dress hat coat
jacket pants car cars bus train plane airplane aeroplane boat ship bike bicycle truck rocket
robot robots music video videos show shows movie cartoon game song

food milk juice tea coffee bread rice roti dal egg eggs cake cookie cookies biscuit biscuits
candy sweets chocolate icecream cream pizza sandwich soup snack breakfast sugar salt fruit
fruits vegetable vegetables potato potatoes onion

tail tails wing wings fur feather feathers paw paws claw claws horn horns trunk beak nest shell spots
stripes dinosaur dinosaurs dragon dragons unicorn unicorns monster monsters fairy fairies ghost ghosts
pirate pirates superhero spider spiders bug bugs insect insects worm worms fish fishes shark sharks
penguin penguins zebra zebras giraffe giraffes kangaroo dolphin dolphins rabbit rabbits turtle turtles
bunny puppy kitten animal pet pets farm zoo
sun moon star stars sky cloud clouds rain snow wind tree trees flower flowers leaf leaves river
sea ocean beach mountain mountains hill forest jungle park rock rocks stone sand fire earth
world space planet planets rainbow weather storm thunder lightning ice

day days night nights morning afternoon evening today tomorrow yesterday week weeks month months
year years time times minute minutes second seconds hour hours moment moments now later soon
early late while

school class lesson lessons homework fun idea ideas word words letter letters number numbers
story stories poem poems joke jokes riddle riddles quiz fact facts thing things stuff way ways
part parts kind sort bit piece place side bottom middle front end start beginning

head hair eye eyes ear ears nose mouth teeth tooth tongue face neck shoulder shoulders arm arms
hand hands finger fingers leg legs knee knees foot feet toe toes tummy belly back

good bad big small little tall short long hot cold warm cool new old young happy sad nice kind
mean pretty beautiful ugly funny silly scary easy hard soft loud quiet fast slow quick high low
full empty dirty wet dry heavy dark bright best better worse worst favourite favorite great
amazing awesome super wonderful lovely cute tiny huge right wrong true false real fake different
same next last first second third fine ready done finished bored tired hungry thirsty sleepy
angry scared afraid excited sick hurt careful clever smart brave strong weak loose tight open
closed sure alone

hello hi hey bye goodbye wow oh ah ha haha yay oops ouch hmm hooray sorry pardon excuse welcome

yourself buddy
"""

COMMON_WORDS: frozenset[str] = frozenset(_WORDS.split())


def is_common_word(tok: str) -> bool:
    """True for everyday English words, including regular -s/-es/-ies/-ed/-ing/-er/-ly forms."""
    if tok in COMMON_WORDS:
        return True
    t = tok[:-2] if tok.endswith("'s") else tok
    cands = [t]
    if t.endswith("ies") and len(t) > 4:
        cands.append(t[:-3] + "y")
    if t.endswith("es") and len(t) > 3:
        cands.append(t[:-2])
    if t.endswith("s") and len(t) > 3:
        cands.append(t[:-1])
    if t.endswith("ed") and len(t) > 4:
        cands += [t[:-2], t[:-1], t[:-3]]  # jumped, liked, stopped
    if t.endswith("ing") and len(t) > 5:
        cands += [t[:-3], t[:-3] + "e", t[:-4]]  # jumping, making, running
    if t.endswith(("er", "ly")) and len(t) > 4:
        cands += [t[:-2], t[:-1]]
    if t.endswith("est") and len(t) > 5:
        cands += [t[:-3], t[:-2]]
    return any(c in COMMON_WORDS for c in cands)
