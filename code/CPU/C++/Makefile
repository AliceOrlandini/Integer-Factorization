CC = g++

CFLAGS  = -std=c++11 -lpthread

TARGET = inefficient_trial_division

all: $(TARGET)
 	$(TARGET): $(TARGET).cpp
		$(CC) -o $(TARGET).exe $(TARGET).cpp $(CFLAGS) 
 
clean:
	$(RM) $(TARGET).exe