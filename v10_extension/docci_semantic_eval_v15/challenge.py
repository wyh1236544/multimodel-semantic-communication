"""Hand-authored semantic perturbations, NOT recovered experimental outputs."""
from common import text_sha

SCENES = [
    ["Two dogs, one black and one white, stand to the left of a red bench. Neither dog wears a collar.",
     "A black dog and a white dog are standing on the left side of a red bench. Both dogs are without collars.",
     "Three dogs, one black and two white, stand to the left of a red bench. None wears a collar.",
     "Two dogs, one black and one white, stand to the left of a blue bench. Neither dog wears a collar.",
     "Two dogs, one black and one red, stand to the left of a white bench. Neither dog wears a collar.",
     "Two dogs, one black and one white, stand to the right of a red bench. Neither dog wears a collar.",
     "Two dogs, one black and one white, stand to the left of a red bench. Both dogs wear collars.",
     "Two dogs, one black and one white, run to the left of a red bench. Neither dog wears a collar.",
     "Two dogs stand near a bench.", "A boat floats on a lake."],
    ["A woman in a blue coat holds two red cups above a table. There is no plate on the table.",
     "Two red cups are held over a table by a woman wearing a blue coat. The table has no plate.",
     "A woman in a blue coat holds three red cups above a table. There is no plate on the table.",
     "A woman in a green coat holds two red cups above a table. There is no plate on the table.",
     "A woman in a red coat holds two blue cups above a table. There is no plate on the table.",
     "A woman in a blue coat holds two red cups below a table. There is no plate on the table.",
     "A woman in a blue coat holds two red cups above a table. There is a plate on the table.",
     "A woman in a blue coat washes two red cups above a table. There is no plate on the table.",
     "A woman is near a table.", "A train passes through a tunnel."],
    ["Three yellow birds sit on a green branch above a wooden fence. No bird is flying.",
     "On a green branch over a wooden fence sit three birds with yellow feathers. None of them is in flight.",
     "Two yellow birds sit on a green branch above a wooden fence. No bird is flying.",
     "Three blue birds sit on a green branch above a wooden fence. No bird is flying.",
     "Three green birds sit on a yellow branch above a wooden fence. No bird is flying.",
     "Three yellow birds sit on a green branch below a wooden fence. No bird is flying.",
     "Three yellow birds sit on a green branch above a wooden fence. One bird is flying.",
     "Three yellow birds peck at a green branch above a wooden fence. No bird is flying.",
     "Birds are near a fence.", "A car is parked inside a garage."],
    ["A boy in a red shirt carries two blue bags to the left of a door. He is not wearing a hat.",
     "To the door's left, a boy wearing a red shirt is carrying a pair of blue bags. His head has no hat.",
     "A boy in a red shirt carries one blue bag to the left of a door. He is not wearing a hat.",
     "A boy in a yellow shirt carries two blue bags to the left of a door. He is not wearing a hat.",
     "A boy in a blue shirt carries two red bags to the left of a door. He is not wearing a hat.",
     "A boy in a red shirt carries two blue bags to the right of a door. He is not wearing a hat.",
     "A boy in a red shirt carries two blue bags to the left of a door. He is wearing a hat.",
     "A boy in a red shirt opens two blue bags to the left of a door. He is not wearing a hat.",
     "A boy is near a door.", "Waves break on an empty beach."]
]
KINDS = ['identity', 'paraphrase', 'count', 'color', 'binding', 'spatial', 'negation', 'action', 'omission', 'unrelated']


def cases():
    result = []
    for i, texts in enumerate(SCENES):
        for kind, candidate in zip(KINDS, texts):
            eid = 'synthetic_scene_{:02d}'.format(i)
            result.append(dict(case_id='challenge/' + kind + '/' + eid, example_id=eid,
                 arm='challenge', control=kind, reference=texts[0], candidate=candidate,
                 reference_sha256=text_sha(texts[0]), candidate_sha256=text_sha(candidate),
                 scope='HAND_AUTHORED_SYNTHETIC_EVALUATOR_DIAGNOSTIC_NOT_MODEL_OUTPUT'))
    return result
