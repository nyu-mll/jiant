from jiant.utils.tokenizers import get_tokenizer, bert_get_tokenized_string_span_map, MosesTokenizer


def test_bert_get_tokenized_string_span_map1():
    text = "What does أنۢبياء anbiyā' mean in English?"
    b_tokenizer = get_tokenizer("bert-large-cased")
    result = bert_get_tokenized_string_span_map(text, b_tokenizer.tokenize(text))
    assert tuple(result) == (
        ("What", 0, 5),
        ("does", 5, 9),
        ("[UNK]", 9, 18),
        ("an", 18, 20),
        ("##bi", 20, 22),
        ("##y", 22, 23),
        ("##ā", 23, 24),
        ("'", 24, 26),
        ("mean", 26, 31),
        ("in", 31, 34),
        ("English", 34, 41),
        ("?", 41, 42),
    )


def test_bert_get_tokenized_string_span_map2():
    text = "What does أنۢبياء أنۢبياء anbiyā' mean in English?"
    b_tokenizer = get_tokenizer("bert-large-cased")
    result = bert_get_tokenized_string_span_map(text, b_tokenizer.tokenize(text))
    assert tuple(result) == (
        ("What", 0, 5),
        ("does", 5, 9),
        ("[UNK]", 9, 26),
        ("[UNK]", 26, 26),
        ("an", 26, 28),
        ("##bi", 28, 30),
        ("##y", 30, 31),
        ("##ā", 31, 32),
        ("'", 32, 34),
        ("mean", 34, 39),
        ("in", 39, 42),
        ("English", 42, 49),
        ("?", 49, 50),
    )


def test_moses_ptb_detokenize():
    # Some of the detokenizations aren't perfect (usually dealing with non-English words, or
    #   problematic punctuation. This suite of text sentences just serves as a check against
    #   changes/improvements, to ensure that we get expected behavior.
    test_cases = [
        [
            "For example , the marine archaean Cenarchaeum symbiosum lives within -LRB- is an endosymbiont of -RRB- the sponge Axinella mexicana .",
            "For example, the marine archaean Cenarchaeum symbiosum lives within (is an endosymbiont of) the sponge Axinella mexicana.",
        ],
        [
            "Investigators are focusing on the ATR 72-600 's engines .",
            "Investigators are focusing on the ATR 72-600's engines.",
        ],
        [
            "In the Netherlands , a low-lying country , dams were often applied to block rivers in order to regulate the water level and to prevent the sea from entering the marsh lands .",
            "In the Netherlands, a low-lying country, dams were often applied to block rivers in order to regulate the water level and to prevent the sea from entering the marsh lands.",
        ],
        [
            "Although wildcats are solitary , the social behavior of domestic cats is much more variable and ranges from widely dispersed individuals to feral cat colonies that form around a food source , based on groups of co-operating females .",
            "Although wildcats are solitary, the social behavior of domestic cats is much more variable and ranges from widely dispersed individuals to feral cat colonies that form around a food source, based on groups of co-operating females.",
        ],
        [
            "This in turn can be made into a nisbah adjective ishtir\u0101k\u012b ` socialist ' , from which an abstract noun ishtir\u0101kiyyah ` socialism ' can be derived .",
            "This in turn can be made into a nisbah adjective ishtir\u0101k\u012b `socialist ', from which an abstract noun ishtir\u0101kiyyah` socialism' can be derived.",
        ],
        [
            "For example , if one car traveling east at 60 km/h passes another car traveling east at 50 km/h , then from the perspective of the slower car , the faster car is traveling east at 60 \u2212 50 = 10 km/h .",
            "For example, if one car traveling east at 60 km/h passes another car traveling east at 50 km/h, then from the perspective of the slower car, the faster car is traveling east at 60 \u2212 50 = 10 km/h.",
        ],
        [
            "In an interview to be aired on Sky News today , he said the housing market is the `` biggest risk '' to the economy and has `` deep , deep structural problems '' .",
            'In an interview to be aired on Sky News today, he said the housing market is the "biggest risk" to the economy and has "deep, deep structural problems".',
        ],
        [
            "The letters were sent to ministers of the previous Labour government in the years of 2004 and 2005 and contain advocacy that has been described as `` particularly frank '' .",
            'The letters were sent to ministers of the previous Labour government in the years of 2004 and 2005 and contain advocacy that has been described as "particularly frank".',
        ],
        [
            "A `` bucky '' lamb is a lamb which was not castrated early enough , or which was castrated improperly -LRB- resulting in one testicle being retained -RRB- .",
            'A "bucky" lamb is a lamb which was not castrated early enough, or which was castrated improperly (resulting in one testicle being retained).',
        ],
        [
            "A significant hurdle in this research is proving that a subject 's conscious mind has not grasped a certain stimulus , due to the unreliability of self-reporting .",
            "A significant hurdle in this research is proving that a subject's conscious mind has not grasped a certain stimulus, due to the unreliability of self-reporting.",
        ],
        [
            "I had the impression he was a serious and conscientious person '' .",
            'I had the impression he was a serious and conscientious person ".',
        ],
        [
            "In his inaugural address , Kennedy made the ambitious pledge to `` pay any price , bear any burden , meet any hardship , support any friend , oppose any foe , in order to assure the survival and success of liberty . ''",
            'In his inaugural address, Kennedy made the ambitious pledge to "pay any price, bear any burden, meet any hardship, support any friend, oppose any foe, in order to assure the survival and success of liberty."',
        ],
        [
            "The first nominee of the Party was John Hospers , who died last year , and he was a supporter of Republican President George W. Bush .",
            "The first nominee of the Party was John Hospers, who died last year, and he was a supporter of Republican President George W. Bush.",
        ],
        [
            "In his view , transactions in a market economy are voluntary , and that the wide diversity that voluntary activity permits is a fundamental threat to repressive political leaders and greatly diminish their power to coerce .",
            "In his view, transactions in a market economy are voluntary, and that the wide diversity that voluntary activity permits is a fundamental threat to repressive political leaders and greatly diminish their power to coerce.",
        ],
        [
            "Resistance organizations in the Gaza Strip continued to launch rockets aimed at the Tel Aviv area and other areas of Israel ; some of these rockets were intercepted by Israel 's Iron Dome system .",
            "Resistance organizations in the Gaza Strip continued to launch rockets aimed at the Tel Aviv area and other areas of Israel; some of these rockets were intercepted by Israel's Iron Dome system.",
        ],
        [
            "Baruch ben Neriah , Jeremiah 's scribe , used this alphabet to create the later scripts of the Old Testament .",
            "Baruch ben Neriah, Jeremiah's scribe, used this alphabet to create the later scripts of the Old Testament.",
        ],
        [
            "To digest vitamin B12 non-destructively , haptocorrin in saliva strongly binds and protects the B12 molecules from stomach acid as they enter the stomach and are cleaved from their protein complexes .",
            "To digest vitamin B12 non-destructively, haptocorrin in saliva strongly binds and protects the B12 molecules from stomach acid as they enter the stomach and are cleaved from their protein complexes.",
        ],
        [
            "We 're a lot stronger than we ever have been , I think , mentally .",
            "We're a lot stronger than we ever have been, I think, mentally.",
        ],
        [
            "Opponents consider governments as agents of neo-colonialism that are subservient to multinational corporations .",
            "Opponents consider governments as agents of neo-colonialism that are subservient to multinational corporations.",
        ],
        [
            "In nouns , inflection for case is required in the singular for strong masculine and neuter nouns , in the genitive and sometimes in the dative .",
            "In nouns, inflection for case is required in the singular for strong masculine and neuter nouns, in the genitive and sometimes in the dative.",
        ],
        [
            "DA Rosen 's November 7 presentation claimed honoring ICE holds would `` produce an undetermined amount of cost savings by reducing probation costs '' , as individuals otherwise on probation would be transferred to federal detention .",
            'DA Rosen\'s November 7 presentation claimed honoring ICE holds would "produce an undetermined amount of cost savings by reducing probation costs", as individuals otherwise on probation would be transferred to federal detention.',
        ],
        [
            "Canberra 's response came from Norwood , who drew a foul and scored a point from a free throw .",
            "Canberra's response came from Norwood, who drew a foul and scored a point from a free throw.",
        ],
        [
            "He displayed an interest in literature from a young age , and began reading Greek and Roman myths and the fables of the Grimm brothers which `` instilled in him a lifelong affinity with Europe '' .",
            'He displayed an interest in literature from a young age, and began reading Greek and Roman myths and the fables of the Grimm brothers which "instilled in him a lifelong affinity with Europe".',
        ],
        [
            "In the history of India-Pakistan bilateral relations , the leader of one country has not visited the swearing-in ceremony of a leader of the other since 1947 , when the two countries became independent .",
            "In the history of India-Pakistan bilateral relations, the leader of one country has not visited the swearing-in ceremony of a leader of the other since 1947, when the two countries became independent.",
        ],
        [
            "On Wednesday open-access journal ZooKeys published their paper on two of the new species , Sturnira bakeri and Sturnira burtonlimi .",
            "On Wednesday open-access journal ZooKeys published their paper on two of the new species, Sturnira bakeri and Sturnira burtonlimi.",
        ],
        [
            "These groups often argue for the recognition of obesity as a disability under the US Americans With Disabilities Act -LRB- ADA -RRB- .",
            "These groups often argue for the recognition of obesity as a disability under the US Americans With Disabilities Act (ADA).",
        ],
        [
            "An art car , featuring what might be the largest collection of singing robotic lobsters anywhere in the world was on display , curiously titled the `` Sashimi Tabernacle Choir . ''",
            'An art car, featuring what might be the largest collection of singing robotic lobsters anywhere in the world was on display, curiously titled the "Sashimi Tabernacle Choir."',
        ],
        [
            "As with many of Rio de Janeiro 's cultural monuments , the library was originally off-limits to the general public .",
            "As with many of Rio de Janeiro's cultural monuments, the library was originally off-limits to the general public.",
        ],
        [
            "The initial condition and the final condition of the system are respectively described by values in a configuration space , for example a position space , or some equivalent space such as a momentum space .",
            "The initial condition and the final condition of the system are respectively described by values in a configuration space, for example a position space, or some equivalent space such as a momentum space.",
        ],
        [
            "Weber began his studies of the subject in The Protestant Ethic and the Spirit of Capitalism , in which he argued that the redefinition of the connection between work and piety in Protestantism and especially in ascetic Protestant denominations , particularly Calvinism , shifted human effort towards rational efforts aimed at achieving economic gain .",
            "Weber began his studies of the subject in The Protestant Ethic and the Spirit of Capitalism, in which he argued that the redefinition of the connection between work and piety in Protestantism and especially in ascetic Protestant denominations, particularly Calvinism, shifted human effort towards rational efforts aimed at achieving economic gain.",
        ],
        [
            "Under the assumption of perfect competition , supply is determined by marginal cost .",
            "Under the assumption of perfect competition, supply is determined by marginal cost.",
        ],
        [
            "De Beauvoir 's adopted daughter and literary heir Sylvie Le Bon , unlike Elka\u00efm , published de Beauvoir 's unedited letters to both Sartre and Algren .",
            "De Beauvoir's adopted daughter and literary heir Sylvie Le Bon, unlike Elka\u00efm, published de Beauvoir's unedited letters to both Sartre and Algren.",
        ],
        [
            "There were some residents in the damaged house who were not accounted for until about 4:30 p.m. when emergency personnel confirmed that three people inside the house were also killed .",
            "There were some residents in the damaged house who were not accounted for until about 4:30 p.m. when emergency personnel confirmed that three people inside the house were also killed.",
        ],
        [
            "`` I think from the very beginning , one of the challenges we 've had with Iran is that they have looked at this administration and felt that the administration was not as strong as it needed to be .",
            "\"I think from the very beginning, one of the challenges we've had with Iran is that they have looked at this administration and felt that the administration was not as strong as it needed to be.",
        ],
        [
            "Ricardo Men\u00e9ndez , vice president of the Productive Economic Area , said Venezuelan President Hugo Ch\u00e1vez has yearned for the creation of this project to empower Venezuelan construction .",
            "Ricardo Men\u00e9ndez, vice president of the Productive Economic Area, said Venezuelan President Hugo Ch\u00e1vez has yearned for the creation of this project to empower Venezuelan construction.",
        ],
        [
            "In ancient Roman culture , Sunday was the day of the Sun god .",
            "In ancient Roman culture, Sunday was the day of the Sun god.",
        ],
        [
            "According to him from his autobiography `` Tezkiret\u00fc ' l B\u00fcnyan '' , his masterpiece is the Selimiye Mosque in Edirne .",
            'According to him from his autobiography "Tezkiret\u00fc \'l B\u00fcnyan", his masterpiece is the Selimiye Mosque in Edirne.',
        ],
        [
            "These groups appear to have had a common origin with Viridiplantae and the three groups form the clade Archaeplastida , whose name implies that their chloroplasts were derived from a single ancient endosymbiotic event .",
            "These groups appear to have had a common origin with Viridiplantae and the three groups form the clade Archaeplastida, whose name implies that their chloroplasts were derived from a single ancient endosymbiotic event.",
        ],
        [
            "Yesterday , the US 's Obama administration said it has ordered an investigation into the appropriateness of military hardware being sold to and deployed by police forces in the United States .",
            "Yesterday, the US's Obama administration said it has ordered an investigation into the appropriateness of military hardware being sold to and deployed by police forces in the United States.",
        ],
        [
            "The five primary classifications can be further divided into secondary classifications such as rain forest , monsoon , tropical savanna , humid subtropical , humid continental , oceanic climate , Mediterranean climate , desert , steppe , subarctic climate , tundra , and polar ice cap .",
            "The five primary classifications can be further divided into secondary classifications such as rain forest, monsoon, tropical savanna, humid subtropical, humid continental, oceanic climate, Mediterranean climate, desert, steppe, subarctic climate, tundra, and polar ice cap.",
        ],
        [
            "Some of the pictures and video coming out of the country do n't show blood and dismembered bodies , but instead show , according to witnesses , those who died from apparent suffocation ; some were foaming at the mouth and others were having convulsions .",
            "Some of the pictures and video coming out of the country do n't show blood and dismembered bodies, but instead show, according to witnesses, those who died from apparent suffocation; some were foaming at the mouth and others were having convulsions.",
        ],
        [
            "Its glyphs were formed by pressing the end of a reed stylus into moist clay , not by tracing lines in the clay with the stylus as had been done previously .",
            "Its glyphs were formed by pressing the end of a reed stylus into moist clay, not by tracing lines in the clay with the stylus as had been done previously.",
        ],
        [
            "In its modern form , the Greek language is the official language in two countries , Greece and Cyprus , a recognised minority language in seven other countries , and is one of the 24 official languages of the European Union .",
            "In its modern form, the Greek language is the official language in two countries, Greece and Cyprus, a recognised minority language in seven other countries, and is one of the 24 official languages of the European Union.",
        ],
        [
            "For example , several writers in the early 1970s used the term to describe fax document transmission .",
            "For example, several writers in the early 1970s used the term to describe fax document transmission.",
        ],
        [
            "Their lawyer Paul Castillo said today the couple `` knew that by coming forward they could help accelerate equality for all same-sex couples in Indiana by demonstrating the urgency of their need for equal dignity . ''",
            'Their lawyer Paul Castillo said today the couple "knew that by coming forward they could help accelerate equality for all same-sex couples in Indiana by demonstrating the urgency of their need for equal dignity."',
        ],
        [
            "For example , from the basic root sh-r-k ` share ' can be derived the Form VIII verb ishtaraka ` to cooperate , participate ' , and in turn its verbal noun ishtir\u0101k ` cooperation , participation ' can be formed .",
            "For example, from the basic root sh-r-k `share 'can be derived the Form VIII verb ishtaraka` to cooperate, participate', and in turn its verbal noun ishtir\u0101k `cooperation, participation 'can be formed.",
        ],
        [
            "Russia Today goes on to say these laws are , `` intend -LSB- ed -RSB- to keep minors from being influenced by non-traditional sexual relationship propaganda and it will be enforced with fines , but not criminal punishment . ''",
            'Russia Today goes on to say these laws are, "intend [ed] to keep minors from being influenced by non-traditional sexual relationship propaganda and it will be enforced with fines, but not criminal punishment."',
        ],
        [
            "The language is spoken by at least 13 million people today in Greece , Cyprus , Italy , Albania , Turkey , and the Greek diaspora .",
            "The language is spoken by at least 13 million people today in Greece, Cyprus, Italy, Albania, Turkey, and the Greek diaspora.",
        ],
        [
            "She spurred on attendees of the event to become more active in supporting women 's rights : `` It 's time for young women in this country to join the fight , because it 's our rights and our health that are at stake . ''",
            "She spurred on attendees of the event to become more active in supporting women's rights: \"It's time for young women in this country to join the fight, because it's our rights and our health that are at stake.\"",
        ],
        [
            "Rajoy is to visit the site of the accident today .",
            "Rajoy is to visit the site of the accident today.",
        ],
        [
            "From Southwest Asia domestic dairy animals spread to Europe -LRB- beginning around 7000 BC but not reaching Britain and Scandinavia until after 4000 BC -RRB- , and South Asia -LRB- 7000 -- 5500 BC -RRB- .",
            "From Southwest Asia domestic dairy animals spread to Europe (beginning around 7000 BC but not reaching Britain and Scandinavia until after 4000 BC), and South Asia (7000 -- 5500 BC).",
        ],
    ]
    moses_tokenizer = MosesTokenizer()
    for sent, detok_sent_gold in test_cases:
        tokens = sent.split()
        detok_sent = moses_tokenizer.detokenize_ptb(tokens)
        assert detok_sent == detok_sent_gold
