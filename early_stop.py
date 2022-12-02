def early_stopping(cfg, trainer, patience):
    '''     Assumptions:
    - The test data is registered in the DatasetCatalog as "custom_test"
    
    Flow:
    1. Evaluate model and get AP.
    2. Save the AP to disk only it is currently the best.
    3. On successive iterations, get previous stats (AP) from disk.
    4. Compare the new_AP with the previous_AP.
    5. If new_AP is same or less than previous_AP, wait for patience_iters and stop.
    6. If the new_AP is an improvement, we reset patience_iters = 0.
    7. If patience_iters == patience, we save the model in output_dir as model_final.pth and stop.
    '''

    # Calculate accuracy/AP
    cfg.DATASETS.TEST = ("val",)
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=cfg.OUTPUT_DIR)
    results = trainer.test(cfg, trainer.model, evaluators = [evaluator])
    
    ## get new AP for bbox or segm
    new_AP = results['bbox']['AP50']

    # If new AP50 is "nan", it means the model has not learned anything, so we just return to training loop
    if np.isnan(new_AP):
        return
    
    model_file_name = "best_model.pth"
    
    # If best model file does not exist, save current as best model
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, model_file_name)) == False:
        torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, model_file_name))

    # current stats
    obj = {
      'model_name': model_file_name,
         'AP50': new_AP
    }
    
    # check if there is a history of accuracies by checking if the file exists
    file_name = 'last_best_acc.json'
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, file_name)):
      
        # read previous stats
        with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'r') as f:
            previous_stats = json.load(f)

        # get previous accuracy
        previous_AP = previous_stats['AP50']
        previous_model_file_name = previous_stats['model_name']

        # if new accuracy is less than previous accuracy, wait and stop!!
        if new_AP <= previous_AP:
            # if patience iter is reached, save model and stop
            if trainer.patience_iter == cfg.SOLVER.PATIENCE:
                # rename best_model.pth to model_final.pth and stop training
                os.rename(os.path.join(cfg.OUTPUT_DIR, previous_model_file_name),
                     os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
                return True
            trainer.patience_iter += 1

        else: # continue training
            # reset patience_iter
            trainer.patience_iter = 0
            
            # save as best_model.pth
            print("Saving current model as best model")
            torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, model_file_name))

            # write current stats to disk
            with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w') as f:
                json.dump(obj, f)

    # writing first evaluation stats to disk	          
    else: 
        with open(os.path.join(cfg.OUTPUT_DIR, file_name), 'w') as f:
            json.dump(obj, f)