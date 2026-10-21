"""State continuity checks using random weights, independent of outputs/."""

import pytest
import torch

from inference.session import SequenceSession
from inference.stream import LTCStream
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier


@torch.no_grad()
def test_blocks_samples_and_session_agree():
    torch.manual_seed(42)
    model = SequenceClassifier(LTCConfig(3, 4)).eval()
    samples = torch.randn(9, 3)
    expected = model(samples.unsqueeze(0), dt=0.1)

    blocks = LTCStream(model, dt=0.1)
    blocks.push_block(samples[:4])
    torch.testing.assert_close(blocks.push_block(samples[4:]), expected)

    session = SequenceSession(LTCStream(model, dt=0.1), samples)
    first = session.advance()
    first_state = first.state.clone()
    first.state.fill_(999)
    torch.testing.assert_close(session.current().state, first_state)
    assert session.processed == 1
    assert session.current().time == 0.1

    while not session.finished:
        session.advance()
    final = session.current()
    torch.testing.assert_close(final.output, expected[0])
    torch.testing.assert_close(final.state, blocks.state_snapshot()[0])
    with pytest.raises(StopIteration):
        session.advance()
    assert session.processed == len(samples)

    session.reset()
    assert session.current() is None
    assert session.processed == 0
    torch.testing.assert_close(session.advance().state, first_state)
