CC = g++

CFLAGS  = -std=c++11 -lpthread

TARGET = inefficient_trial_division

all: $(TARGET)
 	$(TARGET): $(TARGET).cpp
		$(CC) $(CFLAGS) -o $(TARGET).exe $(TARGET).cpp
 
clean:
	$(RM) $(TARGET).exe