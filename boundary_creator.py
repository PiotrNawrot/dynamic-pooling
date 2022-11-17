import torch
import sentencepiece as spm


class BoundaryCreator():
    def __init__(
        self,
        boundaries_type,
        fixed_sf,
        whitespace_id,
        **kwargs,
    ):
        self.boundaries_type = boundaries_type
        self.whitespace_id = whitespace_id

        if boundaries_type == 'fixed':
            assert fixed_sf > 0
            self.fixed_sf = fixed_sf

    def get_boundaries(self, txt=None, tensor=None):
        """
            Function that generates boundaries for given tensor of data

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """
        assert tensor is not None
        data = tensor

        data = data.transpose(0, 1)  # batch_size x seq_len
        boundaries = torch.zeros_like(data, dtype=torch.bool)

        if self.boundaries_type == 'whitespaces':
            boundaries |= (data == self.whitespace_id)
        elif self.boundaries_type == 'fixed':
            boundaries[:, ::self.fixed_sf] = 1
        else:
            return None

        return boundaries


class SPMBoundaries():
    def __init__(self, tokenizer_path, **kwargs):
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    def get_boundaries(self, txt=None, tensor=None):
        """
            This boundaries are compatible with boundary ends group
            For each modality/dataset it's worth to investigate produced
            boundaries with debugger

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """
        assert txt is not None
        data = txt

        words_set = set()
        batch_size = len(data)

        for i in range(batch_size):
            words_set.update(data[i].split(' '))

        words_list = list(words_set)

        words_segmentation = {}

        for word, segmentation in zip(words_list,
                                      self.tokenizer.encode(words_list,
                                                            out_type=str)):
            if word == '':
                words_segmentation[''] = [0]
                continue
            else:
                assert len(segmentation)
                assert len(segmentation[0])
                assert segmentation[0].startswith('▁')

            if segmentation[0] == '▁':
                segmentation = segmentation[1:]
            else:
                segmentation[0] = segmentation[0][1:]

            words_segmentation[word] = [len(x) for x in segmentation]
            assert len(word) == sum(words_segmentation[word])

        sample_lengths = []

        for i in range(batch_size):
            words_lengths = [words_segmentation[word] for word in data[i].split(" ")]
            pieces_lengths = [
                ((y + 1) if (i > 0 and j == (len(sublengths) - 1)) else y)
                for i, sublengths in enumerate(words_lengths)
                for j, y in enumerate(sublengths)
            ]
            sample_lengths.append(torch.tensor(pieces_lengths))

        total_lengths = [x.sum().item() for x in sample_lengths]
        assert len(set(total_lengths)) == 1
        assert total_lengths[0] == len(data[0])
        boundaries = torch.zeros(batch_size, total_lengths[0])

        for i in range(batch_size):
            boundaries[i, sample_lengths[i].cumsum(dim=0)[:-1]] = 1

        return boundaries


def get_boundary_creator(boundaries_type, **kwargs):
    if boundaries_type == 'unigram':
        return SPMBoundaries(**kwargs)
    else:
        return BoundaryCreator(boundaries_type, **kwargs)


if __name__ == '__main__':
    pass
