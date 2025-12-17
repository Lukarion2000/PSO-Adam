import tensorflow as tf
import numpy as np


#############
# PSOModel Class
#############

class PSOModel(tf.keras.Model):
    """
    Custom Keras Model that uses PSO for optimization.
    This model will contain the weights of the best PSO agent after each Trainingstep is done.
    Agent models should not be created via this class, but rather created seperately with the same structure as the main model.
    To access the agent models, use the .models attribute.
    """
    def __init__(self, num_agents=10, agent_batch_size=128, pso_interval=1, c1=0.5, c2=0.5, custom_callbacks = None, *args, **kwargs):
        super(PSOModel, self).__init__(*args, **kwargs)
        self.num_agents = int(num_agents)
        self.agent_batch_size = agent_batch_size
        self.pso_interval = max(pso_interval, 1)
        self.c1 = c1
        self.c2 = c2
        self.pso_counter = 0
        self.skip_pso = True
        self.models = []
        self.agents = []
        self.best_agent_index = 0

        self.agent_batch_size = agent_batch_size
        self.pso_interval = max(pso_interval,1) #pso_interval must be atleast 1
        self.pso_counter = 0
        self.skip_pso = True

        # Arrays für die Differenzen der Kernels und Biases zum besten Agenten
        self.social_kernel_diffs = [None] * num_agents
        self.social_bias_diffs = [None] * num_agents
        
        # Arrays für die Differenzen der Kernels und Biases Personal Best
        self.cognitive_kernel_diffs = [None] * num_agents
        self.cognitive_bias_diffs = [None] * num_agents

        self.personal_best_loss = [float('inf')] * num_agents
        self.personal_best_kernels = [None] * num_agents
        self.personal_best_biases = [None] * num_agents
        
        self.r1 = [0] * num_agents
        self.r2 = [0] * num_agents

        self.best_agent_index = 0

        # old PSOmodel parameters
        self.loss_fn = None
        self.metrics_list = None
        self.custom_callbacks = custom_callbacks


    def _clone_and_randomize_model(self, model):
        """
        Clones a model and randomizes die weights.
        """
        # Modell kopieren
        new_model = tf.keras.models.clone_model(model)
        new_model.build(model.input_shape)
        return new_model

    def _set_start_models(self, models, loss, metrics):
        """
        Set starting Models.
        Weights of given models will get copied while those for any missing models get randomized.
        atleast one model must be given.
        """
        self.models = []
        if len(models) == 0:
            raise ValueError("Die Liste der Modelle ist leer.")
        for i in range(min(len(models), self.num_agents)):
            n_model = tf.keras.models.clone_model(models[i])
            n_model.build(models[i].input_shape)
            n_model.set_weights(models[i].get_weights())
            self.models.append(n_model)  # Liste von Modellen, die den Agenten zugeordnet sind
        for _ in range(max(self.num_agents-len(models),0)):
            self.models.append(self._clone_and_randomize_model(models[0]))  # eigene Kopie des Modells für jeden Agenten erstellen
            
        for i, new_model in enumerate(self.models):
            new_model.compile(optimizer=self.agents[i], loss=loss, metrics=metrics)

    def get_agent_batch(self, x_batch, y_batch, agent_index, num_agents, agent_batch_size, current_batch_size):
        """
        Return batch for the given agent according to the current batch and the agent batch size.
        An agent_batch_size bigger than the current batch size will return the whole batch,
        just like if agent_batch_size was equal to the current batch size.
        """
        # If agent_batch_size is larger than or equal to the current batch_size, each agent gets the whole batch
        if agent_batch_size >= current_batch_size:
            return x_batch, y_batch
        total_needed = num_agents * agent_batch_size
        # If there are more 
        # Example with agent_batch_size 3: Agent 0 gets 0, 1, 2, Agent 1 gets 3, 4, 5 ...
        if total_needed <= current_batch_size:
            # no overlap between agent batches
            start = agent_index * agent_batch_size
            end = start + agent_batch_size
            return x_batch[start:end], y_batch[start:end]
        else:
            #overlap
            start = (agent_index * agent_batch_size) % current_batch_size
            agent_indices = [(start + i) % current_batch_size for i in range(agent_batch_size)]
            return tf.gather(x_batch, agent_indices), tf.gather(y_batch, agent_indices)

    def compute_losses(self, models, lossfn, x_train, y_train):
        """
        Compute losses for all models on the given data.
        """
        losses = []
        loss_fn = tf.keras.losses.get(lossfn)
        for model in models:
            y_pred = model(x_train, training=True)
            loss = loss_fn(y_train, y_pred)
            losses.append(loss)
        return losses

    def update_personal_best(self, agent_index, loss_value):
        """
        Update the personal best for the given agent. (used for the cognitive component, the best will be used for the Social Component)
        """
        pb_kernels = []
        pb_biases = []
        for layer in self.models[agent_index].layers:
                if hasattr(layer, 'kernel'):
                    pb_kernels.append(layer.kernel.numpy().copy())
                else:
                    pb_kernels.append(None)
                if hasattr(layer, 'bias'):
                    pb_biases.append(layer.bias.numpy().copy())
                else:
                    pb_biases.append(None)
        #Save biases and kernels for distance calculations
        self.personal_best_kernels[agent_index] = pb_kernels
        self.personal_best_biases[agent_index] = pb_biases
        # save the loss value for quick comparison
        self.personal_best_loss[agent_index] = loss_value
        
    # Things like sampleweights are not supported here, yet
    def compare_agents(self, loss_fn, x_train, y_train):
        """
        Compare agent for global best and update personal bests.
        """
        best_loss = self.personal_best_loss[self.best_agent_index]
        losses = self.compute_losses(self.models, loss_fn, x_train, y_train)
        for i, agents in enumerate(self.agents):
            loss = losses[i]
            loss_value = tf.reduce_mean(loss)
            if loss_value < best_loss:
                best_loss = loss_value # new best loss, for the other agents to beat
                self.best_agent_index = i
            # Update personal best if loss is better than personal best
            if loss_value < self.personal_best_loss[i]:
                self.update_personal_best(i, loss_value)

    def calculate_move(self):
        """
        Calculate PSO move for each agent based on the cognitve (personal best) and social components (global best).
        """
        if self.skip_pso:
            return
        global_best_kernels = self.personal_best_kernels[self.best_agent_index]
        global_best_biases = self.personal_best_biases[self.best_agent_index]
        for i, model in enumerate(self.models):
            if self.c2 != 0:
                ########
                #Social component
                kernel_diffs = []
                bias_diffs = []                
                for global_best_kernel, global_best_bias, layer in zip(global_best_kernels, global_best_biases, model.layers):
                    if hasattr(layer, 'kernel'):
                        kernel_diff = global_best_kernel - layer.kernel.numpy()
                        kernel_diffs.append(kernel_diff)
                    else:
                        kernel_diffs.append([])
                    if hasattr(layer, 'bias'):
                        bias_diff = global_best_bias - layer.bias.numpy()
                        bias_diffs.append(bias_diff)
                    else:
                        bias_diffs.append([])
                self.social_kernel_diffs[i] = kernel_diffs
                self.social_bias_diffs[i] = bias_diffs
                ########

            if self.c1 != 0:
                ########
                #Cognitive Component
                kernel_diffs = []
                bias_diffs = []
                for layer, pb_kernels, pb_biases in zip(model.layers, self.personal_best_kernels[i], self.personal_best_biases[i]):
                    if hasattr(layer, 'kernel'):
                            kernel_diff = pb_kernels - 1 * layer.kernel.numpy()
                            kernel_diffs.append(kernel_diff)
                    else:
                        kernel_diffs.append([])
                    if hasattr(layer, 'bias'):
                        bias_diff = pb_biases - layer.bias.numpy()
                        bias_diffs.append(bias_diff)
                    else:
                        bias_diffs.append([])
                self.cognitive_kernel_diffs[i] = kernel_diffs
                self.cognitive_bias_diffs[i] = bias_diffs
                ########


    def move_agents_towards_best(self):
        """
        Move agents using the calculated distance from calculate move.
        """
        if self.skip_pso:
            return
        for i, model in enumerate(self.models):
            if self.c2 != 0:
                ########
                # Social Move with c2 and r2
                kernel_diffs = self.social_kernel_diffs[i]
                bias_diffs = self.social_bias_diffs[i]
                # Add randomness
                self.r2[i] = tf.random.uniform(shape=[], minval=0, maxval=1)
                for layer, kernel_diff, bias_diff in zip(model.layers, kernel_diffs, bias_diffs):
                    if hasattr(layer, 'kernel'):
                        if layer.kernel.shape == kernel_diff.shape:
                            layer.kernel.assign(layer.kernel + self.c2 * self.r2[i] * kernel_diff)
                        else:
                            raise ValueError(f"In move_agents_towards_best SocialMove: Incompatible kernel shapes in agent {i}: {layer.kernel.shape} vs. {kernel_diff.shape}")
                    if hasattr(layer, 'bias'):
                        if layer.bias.shape == bias_diff.shape:
                            layer.bias.assign(layer.bias + self.c2 * self.r2[i] * bias_diff)
                        else:
                            raise ValueError(f"In move_agents_towards_best SocialMove: Incompatible bias shapes in agent {i}: {layer.bias.shape} vs. {bias_diff.shape}")
                ########
            if self.c1 != 0:
                ########
                # Cognitive Move with c1 and r1
                kernel_diffs = self.cognitive_kernel_diffs[i]
                bias_diffs = self.cognitive_bias_diffs[i]
                # Add randomness
                self.r1[i] = tf.random.uniform(shape=[], minval=0, maxval=1)
                for layer, kernel_diff, bias_diff in zip(model.layers, kernel_diffs, bias_diffs):
                    if hasattr(layer, 'kernel'):
                        if layer.kernel.shape == kernel_diff.shape:
                            layer.kernel.assign(layer.kernel + self.c1 * self.r1[i] * kernel_diff)
                        else:
                            raise ValueError(f"In move_agents_towards_best CognitiveMove: Incompatible kernel shapes in agent {i}: {layer.kernel.shape} vs. {kernel_diff.shape}")
                    if hasattr(layer, 'bias'):
                        if layer.bias.shape == bias_diff.shape:
                            layer.bias.assign(layer.bias + self.c1 * self.r1[i] * bias_diff)
                        else:
                            raise ValueError(f"In move_agents_towards_best CognitiveMove: Incompatible bias shapes in agent {i}: {layer.bias.shape} vs. {bias_diff.shape}")
                
                ########



    def update_main_model(self, main_model):
        """
        Update the main model with the best weights from the best agent.
        """
        global_best_kernels = self.personal_best_kernels[self.best_agent_index]
        global_best_biases = self.personal_best_biases[self.best_agent_index]
        kernel_idx = 0
        bias_idx = 0
        for main_layer in main_model.layers:
            if hasattr(main_layer, 'kernel'):
                global_best_kernel = global_best_kernels[kernel_idx]
                if global_best_kernel is not None and main_layer.kernel.shape == global_best_kernel.shape:
                    main_layer.kernel.assign(global_best_kernel)
            kernel_idx += 1
            if hasattr(main_layer, 'bias'):
                global_best_bias = global_best_biases[bias_idx]
                if global_best_bias is not None and main_layer.bias.shape == global_best_bias.shape:
                    main_layer.bias.assign(global_best_bias)
            bias_idx += 1

    def compile(self, optimizer, loss, metrics=None, model_list = [], **kwargs):
        """
        Compile the PSOModel with given optimizer, loss and metrics.
        The agents get initialized with the models in model_list.
        If less models than agents are given, the rest will be randomized copies of the first model.
        Atleast one model must be given.
        Each agent model will get its own cloned optimizer.
        """
        self.loss_fn = loss
        self.metrics_list = metrics

        # Clone Optmizer for each Agent
        optimizer_class = optimizer.__class__
        optimizer_config = optimizer.get_config()
        self.agents = [optimizer_class.from_config(optimizer_config) for _ in range(self.num_agents)]

        # Set the starting models from model_list and compile them with their respective optimizers
        self._set_start_models(model_list, loss, metrics)

        # run_eagerly must be True for custom train_step to work properly
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True, **kwargs)

    def set_custom_callbacks(self, callbacks=None):
        """
        Set custom callbacks for the model.
        This is used to pass callbacks that the model needs to send data to.
        ()
        """
        self.custom_callbacks = callbacks
        

    def train_step(self, data):
        """
        Custom train step implementing PSO optimization.
        """
        # get training Data
        x_train, y_train = data
        current_batch_size = x_train.shape[0]

        # 
        if hasattr(self, 'custom_callbacks') and self.custom_callbacks is not None:
            for callback in self.custom_callbacks:
                if hasattr(callback, "x_train_batch") and hasattr(callback, "y_train_batch") and hasattr(callback, "get_first_train_batch"):
                    callback.x_train_batch = x_train.numpy() if hasattr(x_train, "numpy") else x_train
                    callback.y_train_batch = y_train.numpy() if hasattr(y_train, "numpy") else y_train
                    
                    # if it is the first batch of the first Epoch then the callback needs to compute the metrics before the Training happens
                    if callback.get_first_train_batch:
                        callback.get_first_train_batch = False
                        callback.compute_metrics(callback.x_train_batch, callback.y_train_batch)
                        
        # Compare all agents on the current batch and calculate the PSO movement before the Agents own optimizer step
        # Check whether PSO should be done or not.
        if not (self.c1 == 0 and self.c2 == 0):
            self.pso_counter += 1
            if self.pso_counter >= self.pso_interval:
                self.pso_counter = 0
                self.skip_pso = False
                #print("####################################")
                #print("PSO wird durchgeführt")
            else:
                self.skip_pso = True
            if not self.skip_pso:
                # Inertia PSO does not need compare Agents nor calculate moveonly needs move_agent_towards_best when pso is skipped
                self.compare_agents(self.loss_fn, x_train, y_train)
                self.calculate_move()

        #
        for i in range(self.num_agents):
            with tf.GradientTape() as tape:
                x_agent, y_agent = self.get_agent_batch(x_train, y_train, i, self.num_agents, self.agent_batch_size, current_batch_size)
                y_pred = self.models[i](x_agent, training=True)
                loss = self.models[i].compute_loss(x_agent, y_agent, y_pred, sample_weight=None, training=True)
            # calculate gradients to be used for current agent
            grads = tape.gradient(loss, self.models[i].trainable_variables)
            grads_and_vars = list(zip(grads, self.models[i].trainable_variables))
            # apply the Agent's optimizer step to current agent
            self.agents[i].apply_gradients(grads_and_vars)
            # move agents accordning to PSO via the calculated distances

        # When Inertia is used, then move will always happen
        # When not using Inertia, then move_agents will skip according to skip_pso
        self.move_agents_towards_best()
        
        # update_main_model with the weights of the best agent should happen regardless whether pso is happening or not
        # For that to happen correctly, compare_agents is needed
        self.compare_agents(self.loss_fn, x_train, y_train)
        self.update_main_model(self)
        # update metrics
        main_y_pred = self(x_train, training=True)
        for metric in self.metrics:
            metric.update_state(y_train, main_y_pred)
        loss_function = tf.keras.losses.get(self.loss_fn)
        main_loss = loss_function(y_train, main_y_pred)
        main_loss = tf.reduce_mean(main_loss)
        result = {m.name: m.result() for m in self.metrics}
        result["loss"] = main_loss
        return result

############################################



################
# Callback for getting all Metrics during Training
################
class PSOModelMetricsCallback(tf.keras.callbacks.Callback):
    """
    Callback to collect metrics during training for PSOModel (ohne separaten Optimizer).
    """
    def __init__(self, x_val, y_val, batch_counter=1, save_weights_biases=True):
        super().__init__()
        #self.model = model
        self.num_agents = None # wird in on_train_begin gesetzt
        self.batch_counter = max(batch_counter, 1)
        self.batch_check = batch_counter - 1
        self.x_val = x_val
        self.y_val = y_val
        self.x_train_batch = None
        self.y_train_batch = None
        self.get_first_train_batch = True # get 
        self.save_weights_biases = save_weights_biases

        # Metrics
        self.r1 = []
        self.r2 = []
        self.best_agent_index = []
        self.losses = []
        self.accuracies = []
        self.train_batch_losses = []
        self.train_batch_accuracies = []
        self.weights_biases = []
        self.batch_kernels = []
        self.batch_biases = []
        self.adam_norms = []
        self.cognitive_norms = []
        self.social_norms = []
        self.pso_norms = []
        self.complete_norms = []
        self.cos_adam_cognitive = []
        self.cos_adam_social = []
        self.cos_adam_pso = []
        self.cos_social_cognitive = []
        self.cos_adam_complete = []
        self.cos_social_complete = []
        self.cos_cognitive_complete = []
        self.cos_pso_complete = []
        self.weight_diameter_to_best = []
        self.weight_diameter_max = []

    def on_train_begin(self, logs=None):
        self.num_agents = self.model.num_agents    

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            epoch_weights_biases = []
            best_agent_loss = float('inf')
            best_agent = 0
            batch_loss, batch_accuracy = self.compute_metrics(self.x_val, self.y_val)
            for i in range(self.num_agents):
                if batch_loss[i] < best_agent_loss:
                    best_agent_loss = batch_loss[i]
                    best_agent = i
                if self.save_weights_biases:
                    epoch_weights_biases.append(self._get_weights_biases(self.model.models[i]))
            self.best_agent_index.append(best_agent)
            self.losses.append(batch_loss)
            self.accuracies.append(batch_accuracy)
            if self.save_weights_biases:
                self.weights_biases.append(epoch_weights_biases)

    def on_train_batch_begin(self, batch, logs=None):
        if self.model.c1 != 0 or self.model.c2 != 0:
            if self.batch_check == self.batch_counter - 1 and self.model.pso_counter == self.model.pso_interval - 1:
                kernels, biases = [], []
                for m in self.model.models:
                    k, b = self._get_kernels_and_biases(m)
                    kernels.append(k)
                    biases.append(b)
                self.batch_kernels.append(kernels)
                self.batch_biases.append(biases)
        else:
            if self.batch_check == self.batch_counter - 1 and self.model.pso_counter == 0:
                kernels, biases = [], []
                for m in self.model.models:
                    k, b = self._get_kernels_and_biases(m)
                    kernels.append(k)
                    biases.append(b)
                self.batch_kernels.append(kernels)
                self.batch_biases.append(biases)

    def on_train_batch_end(self, batch, logs=None):
        batch_losses, batch_accuracies = self.compute_metrics(self.x_train_batch, self.y_train_batch)
        self.train_batch_losses.append(batch_losses)
        self.train_batch_accuracies.append(batch_accuracies)
        batch_losses, batch_accuracies = self.compute_metrics(self.x_val, self.y_val)
        self.losses.append(batch_losses)
        self.accuracies.append(batch_accuracies)
        self.best_agent_index.append(self.model.best_agent_index)

        if not self.model.skip_pso or (self.model.c1 == 0 and self.model.c2 == 0):
            self.batch_check += 1
            if self.batch_check >= self.batch_counter:
                self.batch_check = 0
                kernels, biases = [], []
                for m in self.model.models:
                    k, b = self._get_kernels_and_biases(m)
                    kernels.append(k)
                    biases.append(b)
                self.batch_kernels.append(kernels)
                self.batch_biases.append(biases)

                batch_r1 = list(self.model.r1)
                batch_r2 = list(self.model.r2)
                self.r1.append(batch_r1)
                self.r2.append(batch_r2)

                self.adam_norms.append([])
                self.social_norms.append([])
                self.cognitive_norms.append([])
                self.pso_norms.append([])
                self.complete_norms.append([])
                self.cos_adam_social.append([])
                self.cos_adam_cognitive.append([])
                self.cos_adam_pso.append([])
                self.cos_social_cognitive.append([])
                self.cos_adam_complete.append([])
                self.cos_social_complete.append([])
                self.cos_cognitive_complete.append([])
                self.cos_pso_complete.append([])

                for agent in range(self.num_agents):
                    # Norm- und Cosine-Berechnungen wie gehabt
                    if len(self.batch_kernels) >= 2:
                        complete_move = self.flatten_and_concat([
                            (np.array(w2) - np.array(w1)) if w1 is not None and w2 is not None else np.zeros_like(w2)
                            for w1, w2 in zip(self.batch_kernels[-2][agent], self.batch_kernels[-1][agent])
                        ] + [
                            (np.array(b2) - np.array(b1)) if b1 is not None and b2 is not None else np.zeros_like(b2)
                            for b1, b2 in zip(self.batch_biases[-2][agent], self.batch_biases[-1][agent])
                        ])
                        complete_norm = np.linalg.norm(complete_move)
                    else:
                        complete_move = None
                        complete_norm = None

                    if self.model.c2 != 0 and self.model.social_kernel_diffs[agent] is not None:
                        social_move = self.flatten_and_concat([
                            np.array(diff) * float(self.model.c2) * float(self.model.r2[agent])
                            if diff is not None and not isinstance(diff, list) else np.zeros(1)
                            for diff in self.model.social_kernel_diffs[agent]
                        ] + [
                            np.array(diff) * float(self.model.c2) * float(self.model.r2[agent])
                            if diff is not None and not isinstance(diff, list) else np.zeros(1)
                            for diff in self.model.social_bias_diffs[agent]
                        ])
                        social_norm = np.linalg.norm(social_move)
                    else:
                        social_move = None
                        social_norm = None

                    if self.model.c1 != 0 and self.model.cognitive_kernel_diffs[agent] is not None:
                        cognitive_move = self.flatten_and_concat([
                            np.array(diff) * float(self.model.c1) * float(self.model.r1[agent])
                            if diff is not None and not isinstance(diff, list) else np.zeros(1)
                            for diff in self.model.cognitive_kernel_diffs[agent]
                        ] + [
                            np.array(diff) * float(self.model.c1) * float(self.model.r1[agent])
                            if diff is not None and not isinstance(diff, list) else np.zeros(1)
                            for diff in self.model.cognitive_bias_diffs[agent]
                        ])
                        cognitive_norm = np.linalg.norm(cognitive_move)
                    else:
                        cognitive_move = None
                        cognitive_norm = None

                    if (self.model.c1 != 0 or self.model.c2 != 0) and (social_move is not None or cognitive_move is not None):
                        if social_move is None:
                            social_move = np.zeros_like(cognitive_move)
                        if cognitive_move is None:
                            cognitive_move = np.zeros_like(social_move)
                        pso_move = social_move + cognitive_move
                        pso_norm = np.linalg.norm(pso_move)
                    else:
                        pso_move = None
                        pso_norm = None

                    if complete_move is not None:
                        if pso_move is not None:
                            adam_move = complete_move - pso_move
                            adam_norm = np.linalg.norm(adam_move)
                        else:
                            adam_move = complete_move
                            adam_norm = complete_norm
                    else:
                        adam_move = None
                        adam_norm = None

                    cos_adam_social = self.cosine_similarity(adam_move, social_move) if (adam_move is not None and social_move is not None) else None
                    cos_adam_cognitive = self.cosine_similarity(adam_move, cognitive_move) if (adam_move is not None and cognitive_move is not None) else None
                    cos_adam_pso = self.cosine_similarity(adam_move, pso_move) if (adam_move is not None and pso_move is not None) else None
                    cos_social_cognitive = self.cosine_similarity(social_move, cognitive_move) if (social_move is not None and cognitive_move is not None) else None
                    cos_adam_complete = self.cosine_similarity(adam_move, complete_move) if (adam_move is not None and complete_move is not None) else None
                    cos_social_complete = self.cosine_similarity(social_move, complete_move) if (social_move is not None and complete_move is not None) else None
                    cos_cognitive_complete = self.cosine_similarity(cognitive_move, complete_move) if (cognitive_move is not None and complete_move is not None) else None
                    cos_pso_complete = self.cosine_similarity(pso_move, complete_move) if (pso_move is not None and complete_move is not None) else None

                    self.adam_norms[-1].append(adam_norm)
                    self.social_norms[-1].append(social_norm)
                    self.cognitive_norms[-1].append(cognitive_norm)
                    self.pso_norms[-1].append(pso_norm)
                    self.complete_norms[-1].append(complete_norm)
                    self.cos_adam_social[-1].append(cos_adam_social)
                    self.cos_adam_cognitive[-1].append(cos_adam_cognitive)
                    self.cos_adam_pso[-1].append(cos_adam_pso)
                    self.cos_social_cognitive[-1].append(cos_social_cognitive)
                    self.cos_adam_complete[-1].append(cos_adam_complete)
                    self.cos_social_complete[-1].append(cos_social_complete)
                    self.cos_cognitive_complete[-1].append(cos_cognitive_complete)
                    self.cos_pso_complete[-1].append(cos_pso_complete)

                agent_vectors = []
                for agent in range(self.num_agents):
                    kernel_vec = self.flatten_and_concat(self.batch_kernels[-1][agent])
                    bias_vec = self.flatten_and_concat(self.batch_biases[-1][agent])
                    agent_vec = np.concatenate([kernel_vec, bias_vec])
                    agent_vectors.append(agent_vec)

                best_idx = self.model.best_agent_index
                best_vec = agent_vectors[best_idx]
                dist_to_best = [np.linalg.norm(vec - best_vec) for vec in agent_vectors]
                max_dist = 0
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        d = np.linalg.norm(agent_vectors[i] - agent_vectors[j])
                        if d > max_dist:
                            max_dist = d

                self.weight_diameter_to_best.append(dist_to_best)
                self.weight_diameter_max.append(max_dist)
                self.batch_kernels = []
                self.batch_biases = []

    def on_epoch_end(self, epoch, logs=None):
        if self.save_weights_biases:
            epoch_weights_biases = []
            for i in range(self.num_agents):
                epoch_weights_biases.append(self._get_weights_biases(self.model.models[i]))
            self.weights_biases.append(epoch_weights_biases)

    def compute_metrics(self, x_train, y_train):
        batch_losses = []
        batch_accuracies = []
        for i in range(self.num_agents):
            y_pred = self.model.models[i](x_train, training=False)
            loss_fn = tf.keras.losses.get(self.model.models[i].loss)
            loss = loss_fn(y_train, y_pred).numpy().mean()
            accuracy = None
            for metric in self.model.metrics_list:
                if "accuracy" in metric.name:
                    metric.reset_state()
                    metric.update_state(y_train, y_pred)
                    accuracy = metric.result().numpy()
                    break
            if accuracy is None:
                metric = self.model.metrics_list[0]
                metric.reset_state()
                metric.update_state(y_train, y_pred)
                accuracy = metric.result().numpy()
            batch_losses.append(loss)
            batch_accuracies.append(accuracy)
        return batch_losses, batch_accuracies

    def _get_weights_biases(self, model):
        weights_biases = []
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights_biases.append(layer.kernel.numpy().copy())
            else:
                weights_biases.append(None)
            if hasattr(layer, 'bias'):
                weights_biases.append(layer.bias.numpy().copy())
            else:
                weights_biases.append(None)
        return weights_biases

    def _get_kernels_and_biases(self, model):
        kernels, biases = [], []
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                kernels.append(layer.kernel.numpy().copy())
            else:
                kernels.append(None)
            if hasattr(layer, 'bias'):
                biases.append(layer.bias.numpy().copy())
            else:
                biases.append(None)
        return kernels, biases

    def flatten_and_concat(self, weight_list):
        flat = []
        for w in weight_list:
            if w is not None and not isinstance(w, list):
                flat.extend(np.array(w).flatten())
            elif isinstance(w, list):
                for item in w:
                    if item is not None:
                        flat.extend(np.array(item).flatten())
        return np.array(flat)

    def cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return np.dot(a, b) / (norm_a * norm_b + 1e-7)