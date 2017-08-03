# tensorprod

Tensorflow is a great framework if you build your models for the purpose of production use:
TF Serving helps you deploy models and serve them with an API ; TF Mobile enables
running TF models in phone apps... But Tensorflow's API can sometimes be a bit coarse.

This library is an attempt to simplify common operations on tensorflow models, in order to
make the job of data scientists more about machine learning and less about industrialization.

With tensorprod, exporting a model in a TF Serving compatible format only takes one method call.
Profiling your graph, writing summaries for tensorboard -- all of this is managed by tensorprod.
Just flip a switch (...inherit from a mixin) and the functionality you want will be available.

Work in progress.

# TODO in this README.md
- [ ] High level description of each subpackage

# TODO in mixit's README.md:
- [ ] Description of the `interfaces` module.
- [ ] List of mixins and their functionality

...
