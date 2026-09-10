# A report on fine-tuning a non-autoregressive image captioning model incorporating a CRF layer, using cross-entropy loss with a repetition-suppressing Viterbi decoding algorithm.

## Introduction

Currently, autoregressive algorithms are used to generate text with generative AI. However, autoregressive algorithms require iterations corresponding to the number of generated tokens, thereby necessitating significant computation time. In contrast, non-autoregressive algorithms do not require iterations based on the word count, allowing for reduced computation time; however, this comes at the cost of observing n-gram repetitions in the generated captions. The model addressed here for non-autoregressive image captioning consists of CLIP, an mlp_connector, BERT, and a CRF layer.

We proposed a Viterbi algorithm that suppresses repetitions.

https://github.com/toshiouchi/stochastic_viterbi_sampling

Generally, for models incorporating a CRF layer, the log-likelihood is calculated using the total score of the correct sentence and the sum of the scores of all possible sentences:

Log-likelihood = -log(total score of the correct sentence / sum of scores of all sentences)

However, since it was unclear whether the loss based on log-likelihood could be calculated in this manner when using the Viterbi algorithm (which suppresses repetitions), I adopted the approach of computing logits and then calculating the cross-entropy loss, similar to standard supervised machine learning.

I will briefly explain the calculation of logits later. Regarding the cross-entropy calculation, since beam approximation is employed, we applied the same approximation to the ground-truth captions; specifically, we excluded any ground-truth captions that fell outside the scope of the beam approximation from the calculation.


## About the Calculation


For the training data, COCO train2017 was used with a split of train:val:test = 0.98:0.01:0.01. The model consists of CLIP, mlp_connector, BERT, and a CRF layer. For training, we first trained a standard image captioning model equipped with a CRF layer, following the method described here: 

https://qiita.com/toshiouchi/items/ee353f0555db7083e259. 

Next, using the parameters from that model, we performed cross-entropy training incorporating Viterbi decoding with repetition suppression. Optuna was used to determine the hyperparameters for this stage of training. During this process, the CLIP and CRF layers were kept frozen (not trained), while the MLP connector, BERT, and other components were trained.

## Evaluation of Calculation Results


Using test data, we calculated the respective scores before and after performing "cross-entropy learning for Viterbi decoding with repetition suppression."

before
```
CIDEr      0.768
rouge-L    0.501
clip_score 0.281
bert_score 0.834
```

after
```
CIDEr      0.828
rouge-L    0.523
clip_score 0.281
bert_score 0.838
```

CIDEr and ROUGE-L have improved. CLIP-Score remains comparable, while BERTScore shows a slight improvement.

### Generated Captions

before
```
hypo: [CLS] a dog laying on a toy mat with its floor. [SEP]
refe: [CLS] a dog chewing a stick laying on the floor. [SEP]
hypo: [CLS] a pair of paper and scissors and a table. [SEP]
refe: [CLS] lots of pairs of scissors sitting on top of a table. [SEP]
hypo: [CLS] a woman holding a pink using under an umbrella. [SEP]
refe: [CLS] a woman uses her mobile phone while holding an umbrella. [SEP]
hypo: [CLS] a large area tower front of a clock. [SEP]
refe: [CLS] tall tower with a large clock in the center of the tower [SEP]
hypo: [CLS] two tour busses parked in a parking lot. [SEP]
refe: [CLS] a couple of large blue buses are parked outside of a building. [SEP]
hypo: [CLS] a person throwing a soccer on a ball. [SEP]
refe: [CLS] a soccer player runs up to kick the ball while the crowd watches. [SEP]
hypo: [CLS] two memorial seating that are benches park bench. [SEP]
refe: [CLS] a couple of creepy statues sitting on a bench in a park at night. [SEP]
hypo: [CLS] a woman in a tennis racquet. [SEP]
refe: [CLS] the female tennis player swings the racket overhead. [SEP]
hypo: [CLS] two giraffes are standing in a grassy field. [SEP]
refe: [CLS] two giraffes standing next to each other in their natural habitat. [SEP]
hypo: [CLS] a bird perched being birds near a window. [SEP]
refe: [CLS] a tall white bird in front of a door next to a small bird. [SEP]
hypo: [CLS] a woman in a baseball bat holding a ball. [SEP]
refe: [CLS] a woman dressed in uniform swinging a baseball bat. [SEP]
hypo: [CLS] a pitcher, catcher players on a baseball game. [SEP]
refe: [CLS] the pitcher just threw the pitch at the baseball game. [SEP]
hypo: [CLS] dogs vintage are in the back of a car truck. [SEP]
refe: [CLS] a truck with a cage in the bed carries live animals on a highway. [SEP]
hypo: [CLS] a batter player swinging a bat at a baseball game. [SEP]
refe: [CLS] the umpire, catcher, and batter playing baseball, just as batter swung his bat. [SEP]
hypo: [CLS] a carrot are prepared laid differentni on top tools let squash fresh
refe: [CLS] there are some fresh vegetables on a cutting board [SEP]
hypo: [CLS] a black and white cat laying on a couch. [SEP]
refe: [CLS] a black brown cat with large whiskers looking at the camera. [SEP]
hypo: [CLS] a man riding a bike on a motorcycle. [SEP]
refe: [CLS] a person sitting at a motorcycle looking at another person. [SEP]
hypo: [CLS] a batter player swinging a bat at a baseball game. [SEP]
refe: [CLS] baseball player completing his batting swing at home plate [SEP]
hypo: [CLS] a man sitting in a on top of boat dog. [SEP]
refe: [CLS] a person sitting on a flat paddle boat with their dog. [SEP]
hypo: [CLS] a bunch of with carrot chopped cutting board multiple knife. [SEP]
refe: [CLS] carrots and cucumber on wooden cutting board near knives. [SEP]
hypo: [CLS] a box made donuts somekin of a table. [SEP]
refe: [CLS] a cook book for making donuts with donuts and coffee pictures on it ' s cover. [SEP]
hypo: [CLS] a bathroom with a wooden in a toilet. [SEP]
refe: [CLS] a white toilet sitting underneath a window next to a tub. [SEP]
hypo: [CLS] a cat sitting on top of a park bench. [SEP]
refe: [CLS] a bird sitting on top of a bench and a cat sitting underneath it. [SEP]
hypo: [CLS] a woman in a pot with a kitchen. [SEP]
refe: [CLS] a person in a room in front of a stove. [SEP]
hypo: [CLS] a woman holding down the street under an umbrella. [SEP]
refe: [CLS] people walking on a city street carrying umbrellas over their heads. [SEP]
hypo: [CLS] a black and white cat laying on a bed. [SEP]
refe: [CLS] a black cat resting atop blankets on a bed [SEP]
hypo: [CLS] a man player is batter to a baseball game. [SEP]
refe: [CLS] batter at home plate watching for the pitch. [SEP]
hypo: [CLS] a man sits in a sitting on a park bench. [SEP]
refe: [CLS] a man on a bench with a view of the city [SEP]
hypo: [CLS] a dog laying tucked colored sitting on a bed. [SEP]
refe: [CLS] a dog is laying on a messed up bed. [SEP]
hypo: [CLS] two men cross country skiing down a hill. [SEP]
refe: [CLS] two skiers with their skis skiing on the slope. [SEP]
hypo: [CLS] several stopeth for directions no signs police arrows go language
refe: [CLS] traffic moving down the road with a street sign on the right. [SEP]
hypo: [CLS] a cat is sitting on top of a ledge. [SEP]
refe: [CLS] a black cat waits inside a metal flowerpot. [SEP]
hypo: [CLS] a castleraf tower front of a clock. [SEP]
refe: [CLS] a large yellow tower with a clock in it [SEP]
hypo: [CLS] this birthday thomas decorated like sitting on a train. [SEP]
refe: [CLS] a train - like birthday cake during a special day. [SEP]
hypo: [CLS] a man oner is jumping in the air snowboard. [SEP]
refe: [CLS] a person flying up into the air after a snowboard jump. [SEP]
hypo: [CLS] a man with a black rain under an umbrella. [SEP]
refe: [CLS] a man running walking in the rain, holding an umbrella. [SEP]
hypo: [CLS] a wooden table topped fruitsies onionsdis vegetables. [SEP]
refe: [CLS] an arrangement of fresh produce including tomatoes, kale, and cabbage on a white tablecloth. [SEP]
hypo: [CLS] a young maner isboard on a skate park. [SEP]
refe: [CLS] a boy is grinding a block on a skateboard [SEP]
hypo: [CLS] a group of people in the on the beach. [SEP]
refe: [CLS] a bunch of kites near the beach with mountains in the background [SEP]
hypo: [CLS] a close up of a pizza on a table. [SEP]
refe: [CLS] a large pizza is on a white plate [SEP]
hypo: [CLS] a young boy swinging a bat holding a baseball game. [SEP]
refe: [CLS] a person taking a swing at a baseball [SEP]
hypo: [CLS] a group of horses on a cart carriage. [SEP]
refe: [CLS] a horse drawn carriage being ridden through a park. [SEP]
hypo: [CLS] a person standing next to a park bench. [SEP]
refe: [CLS] a person with a backpack standing near a large group of empty benches. [SEP]
hypo: [CLS] a woman sitting at a table with a laptop. [SEP]
refe: [CLS] a brunette is at a white table and there is a laptop and cellphone [SEP]
hypo: [CLS] a red hydra engine fire flags in the street. [SEP]
refe: [CLS] the jazz band is taking part in a parade. [SEP]
hypo: [CLS] a red and yellow on a fire hydrant. [SEP]
refe: [CLS] a very cute old looking fire hydrant on the curb. [SEP]
hypo: [CLS] a dog standing next to each other sheep. [SEP]
refe: [CLS] a few sheep are outside in a field with a dog [SEP]
hypo: [CLS] a yellow bus drives down a city street. [SEP]
refe: [CLS] an old green dented bus is making its way along a road. [SEP]
hypo: [CLS] a white topped with a sandwich sitting on a plate. [SEP]
refe: [CLS] the sandwich is on the plate and has been cut in two [SEP]
hypo: [CLS] a piece of cake tea been on a plate. [SEP]
refe: [CLS] cake on white plates and a bottle of milk. [SEP]
hypo: [CLS] an travelers loading people on top of a plane. [SEP]
refe: [CLS] a group of people are headed toward a small aircraft. [SEP]
hypo: [CLS] a woman sitting in front of a laptop computer. [SEP]
refe: [CLS] a women leans over her desk towards a laptop. [SEP]
hypo: [CLS] a man sitting on a couch with a video game. [SEP]
refe: [CLS] two people sitting on a couch and holding game controllers. [SEP]
hypo: [CLS] a cat is playing santa carrot wrapped legs [SEP]  head. [SEP]
refe: [CLS] a cat wearing a holiday hat reclines on an blanket while being petted. [SEP]
hypo: [CLS] a living room and a couch with a table. [SEP]
refe: [CLS] there is a beige couch and orange pillows in this living room [SEP]
hypo: [CLS] a cow is standing in a grassy field. [SEP]
refe: [CLS] native animals lying in grassy field in 3d photograph. [SEP]
hypo: [CLS] a person riding a surfboard on the ocean. [SEP]
refe: [CLS] a person that is on top of a wave. [SEP]
hypo: [CLS] a street sign indicating name locations directionsorted. [SEP]
refe: [CLS] a poll that has several different high school sings on it. [SEP]
hypo: [CLS] a yellow bus drives down a city street. [SEP]
refe: [CLS] a bus on a street with cars and bicyclists. [SEP]
hypo: [CLS] a man is ready cooked on a pizza. [SEP]
refe: [CLS] a man who is putting a pizza in an oven. [SEP]
hypo: [CLS] a group of with boats on the water. [SEP]
refe: [CLS] several boats at a dock near the water. [SEP]
hypo: [CLS] a chineseberriescco om vegetable food. [SEP]
refe: [CLS] a plate and bowl filled with food sitting on a table. [SEP]
hypo: [CLS] a large area tower front of a clock. [SEP]
refe: [CLS] a clock tower rises above the port of san francisco building. [SEP]
hypo: [CLS] a table with pizza on top of a plate. [SEP]
refe: [CLS] a small pizza with several ingredients missing one slice. [SEP]
```

after
```
hypo: [CLS] a dog that is playing its floor with its floor. [SEP]
refe: [CLS] a dog on the floor chewing on a bone. [SEP]
hypo: [CLS] a pair of scissors and scissors and a table. [SEP]
refe: [CLS] a bunch of scissors sitting on a table with a hammer [SEP]
hypo: [CLS] a woman holding a pink using under an umbrella. [SEP]
refe: [CLS] a woman holds up an electronic cigarette underneath an umbrella. [SEP]
hypo: [CLS] a large clock tower front of a clock. [SEP]
refe: [CLS] an older clock tower standing above some buildings. [SEP]
hypo: [CLS] two tour busses parked in a parking lot. [SEP]
refe: [CLS] two blue buses parked in front of a volvo store. [SEP]
hypo: [CLS] a soccer throwing a soccer on a ball. [SEP]
refe: [CLS] a soccer goalie about to kick the ball. [SEP]
hypo: [CLS] two are benches that are benches park bench. [SEP]
refe: [CLS] a couple of creepy statues sitting on a bench in a park at night. [SEP]
hypo: [CLS] a woman in a tennis racquet. [SEP]
refe: [CLS] the attractive tennis player bent low while returning the volley, displaying ample cleavage to the perverted mturk worker who observed her delicate breasts. [SEP]
hypo: [CLS] two giraffes are standing in a grassy field. [SEP]
refe: [CLS] two very pretty giraffes together in a big grassy field. [SEP]
hypo: [CLS] a bird perched being birds near a window. [SEP]
refe: [CLS] two birds are standing outside a sliding glass door [SEP]
hypo: [CLS] a woman in a baseball bat holding a ball. [SEP]
refe: [CLS] a female baseball player unleashes a hit and goes for the run. [SEP]
hypo: [CLS] a baseball game baseball players on a baseball game. [SEP]
refe: [CLS] the pitcher just threw the pitch at the baseball game. [SEP]
hypo: [CLS] two truck are in the back of a car truck. [SEP]
refe: [CLS] a truck with a cage in the bed carries live animals on a highway. [SEP]
hypo: [CLS] a baseball player swinging a bat at a baseball game. [SEP]
refe: [CLS] a group of men in a field playing baseball. [SEP]
hypo: [CLS] a cutting board let laid differentni on top tools let squash fresh
refe: [CLS] an image of a table with vegetables on it [SEP]
hypo: [CLS] a black and white cat laying on a couch. [SEP]
refe: [CLS] a black brown cat with large whiskers looking at the camera. [SEP]
hypo: [CLS] a man on a motorcycle on a motorcycle. [SEP]
refe: [CLS] a man riding on the back of a motorcycle next to another man. [SEP]
hypo: [CLS] a baseball player swinging a bat at a baseball game. [SEP]
refe: [CLS] a baseball player who has just swung his bat at a pitch [SEP]
hypo: [CLS] a man with a dog on a dog boat dog. [SEP]
refe: [CLS] a person that is floating in some water [SEP]
hypo: [CLS] a cutting board with carrot chopped cutting board multiple knife. [SEP]
refe: [CLS] a cutting board with carrots and a cucumber on top. [SEP]
hypo: [CLS] a donuts donuts somekin of a table. [SEP]
refe: [CLS] a cook book for making donuts with donuts and coffee pictures on it ' s cover. [SEP]
hypo: [CLS] a bathroom with a toilet in a toilet. [SEP]
refe: [CLS] a toilet that has cans of paint next to it. [SEP]
hypo: [CLS] a cat sitting on top of a park bench. [SEP]
refe: [CLS] cat underneath park bench with black crow perched on top. [SEP]
hypo: [CLS] a woman in a kitchen with a kitchen. [SEP]
refe: [CLS] a person in a room in front of a stove. [SEP]
```

## How to obtain logits (crf_beam_logits).


Reference Program

```python
    def forward(self, emissions, targets, sampled_beam_idx = None, top_indices = None, grpo_mode = True, crf_mode = False, 
                use_crf_beam_logits = False, masks=None, beam=None):

        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        B = batch_size
        device = emissions.device
        permit_repeat = [ pad_token_id, eos_token_id, cls_token_id, sep_token_id, a_token_id, an_token_id, the_token_id, period_token_id, \
                         comma_token_id, and_token_id, in_token_id, we_token_id, i_token_id, he_token_id, she_token_id, \
                         it_token_id, they_token_id, dbl_token_id, sgl_token_id ]
        
        if top_indices == None:
            beam_emission_scores, beam_targets = torch.topk( emissions, beam, -1)
        else:
            beam_emission_scores = torch.gather( emissions, -1, top_indices )
            beam_targets = top_indices
        
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D
        beam_transition_matrix = torch.bmm(
            beam_transition_score1.view(-1, self.beam, self.rank),
            beam_transition_score2.view(-1, self.beam, self.rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam) # bsz, seq_len, beam, beam

        if not self.ref_t:

            traj_tokens = []
            step_probs = []

            # compute the normalizer in the log-space
            score = beam_emission_scores[:, 0]  # B x K
            #dummy = torch.arange(beam, device=score.device).expand(*score.size()).contiguous()

            logits_t0 = score  / self.temp            
            
            for i in range(1, seq_len):
                _score_matrix = score.unsqueeze(-1) + beam_transition_matrix[:,i-1,:,:,].expand( -1, -1, -1 )
                _score_matrix = _score_matrix + beam_emission_scores[:,i][:,None,:].expand(-1,beam,-1)

                step_probs.append( _score_matrix / self.temp )

                _score2, _index2 = torch.topk( _score_matrix, self.cand, dim = 1 ) 
                _score = _score2[:,0]
                _index = _index2[:,0]
                
                #if masks is not None:
                #    score = torch.where(masks[:, i: i+1], _score, score)
                #    index = torch.where(masks[:, i: i+1], _index, dummy)
                #else:
                score, index = _score, _index
                traj_tokens.append(_index2) # S, B, cand, W

            _, _indexes = torch.topk( score, self.cand, dim = 1 )
            current_sampled_index = _indexes #(B,cand )

            # beam から vocab_size に戻す。
            beam_targets1 = beam_targets[:,-1] # B, W
            current_sampled_index = torch.gather( beam_targets1, -1, current_sampled_index ) #B,cand
        
            finalized_tokens = torch.full( (seq_len,B), self.vocab_size , dtype=torch.long, device=device)
    
            ## 3. 最初の要素として追加
            finalized_tokens[0] = current_sampled_index[:,0] # (S), B

            traj_tokens = torch.stack( traj_tokens, dim = 0 ) # S, B, cand, W
            
            # beam から vocab_size に戻す。
            cand_beam_targets = beam_targets.unsqueeze(2).expand( -1, -1, self.cand, -1 ) # B, seq_len, cand, W
            cand_beam_targets = cand_beam_targets.permute( 1, 0, 2, 3 ) #S,B,cand,W
            cand_beam_targets1 = cand_beam_targets[:-1]
            cand_beam_targets2 = cand_beam_targets[1:]
            traj_tokens = torch.gather( cand_beam_targets1, -1, traj_tokens)
            traj_tokens3 = torch.full( ( seq_len - 1, B, self.cand, self.vocab_size ), self.vocab_size, dtype=torch.long, device = beam_targets.device )
            traj_tokens3 = torch.scatter( traj_tokens3, -1, index = cand_beam_targets2, src = traj_tokens)

            for i3, idx_step in enumerate( torch.flip(traj_tokens3, dims=(0,))):
                i2= i3+1
                previous_pointer = finalized_tokens[i3]
                stop_flag = torch.zeros( (B), dtype=torch.int,device=device) # stop_flag が 0 の時 更新 OK, 1の時、これ以上更新しない。
                for i in range( self.cand ):
                    idx = idx_step[:,i].new_empty((idx_step[:,i].size(0), idx_step[:,i].size(1)+1 ))
                    idx[:,:-1] = idx_step[:,i]
                    idx[:,idx.size(1)-1] = self.vocab_size
                    cand_tokens = torch.gather( idx,-1, previous_pointer.unsqueeze(-1) ).squeeze(-1)# B,N　更新の候補を作成。
                    finalized_tokens2 = finalized_tokens.clone()
                    finalized_tokens2[ i2: ] = -100 # 現在の時刻より先は -100
                    repeat_mask = ( finalized_tokens2 == cand_tokens  )  # S, B 繰り返しの場所を特定する　mask 
                    not_permit_mask = (~torch.isin( finalized_tokens2, torch.tensor( permit_repeat, device=device))).to(torch.int ) 
                    repeat_sum = ( (repeat_mask).to(torch.int) * not_permit_mask ).sum(dim =0 )# B 繰り返しの場所に繰り返しが許可されたトークンの場所をかけて和をとることにより、B の形状の繰り返しがある時1 以上、ない時0 を 得る。
                    if i == self.cand -1:#最終の時は、
                        chg_flag = ~(stop_flag.to(torch.bool)) #最終の前までに stop が1になれば変えない。stop が0だったら変える。
                        if i == 0:
                            chg_flag = torch.ones( (B), dtype=torch.bool,device=device)
                    else:
                        tmp_flag =  stop_flag + repeat_sum  # stopとrepeat両方が0の時 0
                        chg_flag = ( tmp_flag == 0 )# stop と　repeat両方が0の時　chgは　true
                        stop_flag[ stop_flag == 1 ] = 1 # stop_flag が 1の場合は stop_flag =1
                        stop_flag[ chg_flag ] = 1 #chg_flag = True の場合は、更新されたのだから stop_flag = 1
                    finalized_tokens[i2] = torch.where(chg_flag,cand_tokens,finalized_tokens[i2])#(S),B True だったら変更、 False だったらそのまま。
                    chg_flag2 = chg_flag.unsqueeze(1).expand(-1,beam)
            
            finalized_tokens = torch.flip( finalized_tokens, dims= (0, ) )
            finalized_tokens = finalized_tokens.transpose( 0, 1 )#B,S,N

            if use_crf_beam_logits:
                # vocab_size のfinalized_tokens から beam の sampled_beam_idx を作る
                mask = beam_targets == finalized_tokens.unsqueeze(-1) #B,S,W
                # 2. ビーム次元 (-1) で一致しているインデックスを取得
                sampled_beam_idx2 = torch.argmax(mask.to(torch.int32), dim=-1)
        
                beam_logits = []
                #print( "define log_beam_probs None:")
                for i3, probs_step in enumerate( reversed( step_probs )):
                    i4 = seq_len - i3 - 1
                    #current_beam_index = sampled_beam_idx2[:,i4]
                    previous_beam_index = sampled_beam_idx2[:,i4-1]
                    prob_from_prev = probs_step.gather(2, previous_beam_index[:, None, None].expand(-1, 1,  beam))
                    beam_logits.append( prob_from_prev.squeeze(1) )
        
                beam_logits.append( logits_t0 )
                beam_logits = torch.stack( beam_logits, dim = 0 )
                beam_logits = torch.flip( beam_logits, dims = (0, ) )
                beam_logits = beam_logits.permute( 1, 0, 2 )
                crf_beam_logits = beam_logits            
            else:
                crf_beam_logits = torch.tensor( [0] )
    
    return finalized_tokens, crf_beam_logits, beam_targets

```
### Description

Calculate the `_score_matrix` for sequence `i` using the beam approximations for CRF emission probabilities (`beam_emission_scores`) and transition probabilities (`beam_transition_matrix`). By appending `_score_matrix / self.temp`, we obtain a list of tensors—each with shape `(bsz, beam, beam)`—indexed by sequence length (dimension 0). The beam dimension at index 2 corresponds to sequence `i-1` (pre-transition), while the beam dimension at index 3 corresponds to sequence `i` (post-transition). We refer to this as "logits considering sequences `i-1` and `i`." Here, `beam` refers to the `beam_size` used for the approximation in `beam_emission_scores = torch.topk(emissions, beam, -1)`. Meanwhile, we obtain the caption token path `sampled_beam_idx` by following the backtrace of a Viterbi algorithm that accounts for repetition suppression. By applying `sampled_beam_idx` to the `beam` dimension corresponding to sequence `i-1` in the logits tensor of shape `(seq_len, bsz, beam, beam)`, we derive `crf_beam_logits` of shape `(bsz, seq_len, beam)`, representing the logits where sequence `i` is not yet determined.

## Summary

It was found that various scores improved when a Viterbi decoding algorithm incorporating repetition suppression was trained following the supervised machine learning of a model that utilized a CRF layer without repetition suppression.
