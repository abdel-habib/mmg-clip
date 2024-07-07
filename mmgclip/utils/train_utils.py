def epoch_time(start_time, end_time):
    ''''
    Calculates the epoch time given the start and end time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins, elapsed_secs = divmod(elapsed_time, 60)
    return elapsed_mins, elapsed_secs
