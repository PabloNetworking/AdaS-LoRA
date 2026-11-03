import logging

import federatedscope.contrib.common as fs_common
from federatedscope.core.configs import constants

logger = logging.getLogger(__name__)


def get_aggregator(method, model=None, device=None, online=False, config=None):
    """
    This function builds an aggregator, which is a protocol for aggregate \
    all clients' model(s).

    Arguments:
        method: key to determine which aggregator to use
        model:  model to be aggregated
        device: where to aggregate models (``cpu`` or ``gpu``)
        online: ``True`` or ``False`` to use online aggregator.
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        An instance of aggregator (see ``core.aggregator`` for details)

    Note:
      The key-value pairs of ``method`` and aggregators:
        ==================================  ===========================
        Method                              Aggregator
        ==================================  ===========================
        ``tensorflow``                      ``cross_backends.FedAvgAggregator``
        ``local``                           \
        ``core.aggregators.NoCommunicationAggregator``
        ``global``                          \
        ``core.aggregators.NoCommunicationAggregator``
        ``fedavg``                          \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``pfedme``                          \
        ``core.aggregators.ServerClientsInterpolateAggregator``
        ``ditto``                           \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``fedsageplus``                     \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``gcflplus``                        \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``fedopt``                          \
        ``core.aggregators.FedOptAggregator``
        ==================================  ===========================
    """
    if config.backend == 'tensorflow':
        from federatedscope.cross_backends import FedAvgAggregator
        return FedAvgAggregator(model=model, device=device)
    else:
        from federatedscope.core.aggregators import ClientsAvgAggregator, \
            OnlineClientsAvgAggregator, ServerClientsInterpolateAggregator, \
            FedOptAggregator, NoCommunicationAggregator, \
            AsynClientsAvgAggregator, KrumAggregator, \
            MedianAggregator, TrimmedmeanAggregator, \
            BulyanAggregator, NormboundingAggregator, \
            HeteroLoRAAggregator, HetLoRAAggregator, AdaSPruneOnlyAggregator
        from federatedscope.core.aggregators.adas_aggregator import \
            AdaSAggregator
        from federatedscope.core.aggregators.adas_layeraware_aggregator import \
            AdaSLayerAwareAggregator

    STR2AGG = {
        'fedavg': ClientsAvgAggregator,
        'krum': KrumAggregator,
        'median': MedianAggregator,
        'bulyan': BulyanAggregator,
        'trimmedmean': TrimmedmeanAggregator,
        'normbounding': NormboundingAggregator
    }

    method_name = fs_common.normalize_method_name(method)

    if method_name == 'fah_qlora':
        aggregator_type = 'heterolora'
        logger.info(
            "federate.method='%s' uses the shared "
            "HeteroLoRAAggregator underneath.",
            method,
        )
    elif method_name in constants.AGGREGATOR_TYPE:
        aggregator_type = constants.AGGREGATOR_TYPE[method_name]
    else:
        aggregator_type = 'clients_avg'
        logger.warning(
            'Aggregator for method {} is not implemented. Will use default one'
            .format(method))

    if method_name == 'fah_qlora' and fs_common.get_fah_cfg(config) is None:
        logger.warning(
            "federate.method='%s' selected the FAH-QLoRA "
            "worker path, but the active fah config is not enabled.",
            method,
        )

    if config.data.type.lower() == 'hetero_nlp_tasks' and \
            not config.federate.atc_vanilla:
        from federatedscope.nlp.hetero_tasks.aggregator import ATCAggregator
        return ATCAggregator(model=model, config=config, device=device)

    if config.fedopt.use or aggregator_type == 'fedopt':
        return FedOptAggregator(config=config, model=model, device=device)
    elif aggregator_type == 'clients_avg':
        if online:
            return OnlineClientsAvgAggregator(
                model=model,
                device=device,
                config=config,
                src_device=device
                if config.federate.share_local_model else 'cpu')
        elif config.asyn.use:
            return AsynClientsAvgAggregator(model=model,
                                            device=device,
                                            config=config)
        else:
            if config.aggregator.robust_rule not in STR2AGG:
                logger.warning(
                    'The specified %s aggregtion rule has not been supported, '
                    'the vanilla fedavg algorithm will be used instead.',
                    config.aggregator.robust_rule,
                )
            return STR2AGG.get(config.aggregator.robust_rule,
                               ClientsAvgAggregator)(model=model,
                                                     device=device,
                                                     config=config)

    elif aggregator_type == 'server_clients_interpolation':
        return ServerClientsInterpolateAggregator(
            model=model,
            device=device,
            config=config,
            beta=config.personalization.beta)
    elif aggregator_type == 'no_communication':
        return NoCommunicationAggregator(model=model,
                                         device=device,
                                         config=config)
    elif aggregator_type == 'heterolora':
        max_rank = fs_common.get_effective_max_rank(config)
        logger.info(
            "[AggregatorBuilder] Creating HeteroLoRAAggregator with effective max_rank=%s",
            max_rank,
        )
        return HeteroLoRAAggregator(model=model,
                                    device=device,
                                    config=config)
    elif aggregator_type == 'hetlora':
        het_cfg = fs_common.get_hetlora_cfg(config)
        agg_mode = 'sparsity_weighted'
        if het_cfg is not None and hasattr(het_cfg, 'aggregation'):
            agg_mode = getattr(het_cfg.aggregation, 'mode', 'sparsity_weighted')
        max_rank = fs_common.get_effective_max_rank(config)
        logger.info(
            "Creating HetLoRAAggregator with effective max_rank=%s, aggregation_mode='%s'",
            max_rank,
            agg_mode,
        )
        return HetLoRAAggregator(model=model,
                                 device=device,
                                 config=config)
    elif aggregator_type == 'adas_pruneonly':
        ada_cfg = fs_common.get_adas_pruneonly_cfg(config)
        agg_mode = 'sparsity_weighted'
        if ada_cfg is not None and hasattr(ada_cfg, 'aggregation'):
            agg_mode = getattr(ada_cfg.aggregation, 'mode', 'sparsity_weighted')
        max_rank = fs_common.get_effective_max_rank(config)
        logger.info(
            "Creating AdaSPruneOnlyAggregator with effective max_rank=%s, aggregation_mode='%s'",
            max_rank,
            agg_mode,
        )
        return AdaSPruneOnlyAggregator(model=model,
                                       device=device,
                                       config=config)
    elif aggregator_type == 'adas':
        adas_method_cfg = fs_common.get_adas_cfg(config)
        agg_mode = 'sparsity_weighted'
        if adas_method_cfg is not None and hasattr(adas_method_cfg, 'aggregation'):
            agg_mode = getattr(adas_method_cfg.aggregation, 'mode', 'sparsity_weighted')
        max_rank = fs_common.get_effective_max_rank(config)
        logger.info(
            "Creating AdaSAggregator with effective max_rank=%s, aggregation_mode='%s'",
            max_rank,
            agg_mode,
        )
        return AdaSAggregator(model=model,
                                         device=device,
                                         config=config)
    elif aggregator_type == 'adas_layeraware':
        layeraware_cfg = fs_common.get_adas_layeraware_cfg(config)
        agg_mode = 'sample_size'
        if layeraware_cfg is not None and hasattr(layeraware_cfg, 'aggregation'):
            agg_mode = getattr(layeraware_cfg.aggregation, 'mode', 'sample_size')
        max_rank = fs_common.get_effective_max_rank(config)
        logger.info(
            "Creating AdaSLayerAwareAggregator with effective max_rank=%s, aggregation_mode='%s'",
            max_rank,
            agg_mode,
        )
        return AdaSLayerAwareAggregator(model=model,
                                         device=device,
                                         config=config)
    else:
        raise NotImplementedError(
            "Aggregator {} is not implemented.".format(aggregator_type))
