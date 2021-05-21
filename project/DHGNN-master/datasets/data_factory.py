from datasets.cora import load_citation_data, load_cora_data


def source_select(cfg):
    data_type = cfg['data_type']
    standard_split = cfg['standard_split']
    if data_type == 'citation':
        if standard_split:
            a = load_citation_data
            return a
        else:
            return load_cora_data