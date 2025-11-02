from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class ConversationState:
    """Prati stanje trenutne konverzacije."""
    speakers: Set[str] = field(default_factory=set)
    speaker_genders: Dict[str, str] = field(default_factory=dict)
    conversation_type: str = "unknown"  # "female_only", "male_only", "mixed", "unknown"
    last_speaker: Optional[str] = None
    current_dialogue_speakers: Set[str] = field(default_factory=set)
    dialogue_history: List[Tuple[str, str, str]] = field(default_factory=list)  # [(speaker, gender, text), ...]

class ConversationAnalyzer:
    """Analizira konverzacije između više govornika."""
    
    def __init__(self):
        self.conversation_state = ConversationState()
        self.dialogue_window = 5  # Koliko replika unazad gledamo za kontekst
        
    def add_speaker_line(self, speaker: str, gender: Optional[str], text: str):
        """Dodaje novu repliku u konverzaciju."""
        if speaker:
            self.conversation_state.speakers.add(speaker)
            self.conversation_state.current_dialogue_speakers.add(speaker)
            
            if gender:
                self.conversation_state.speaker_genders[speaker] = gender
            
            # Dodaj u povijest dijaloga
            self.conversation_state.dialogue_history.append((speaker, gender or "unknown", text))
            
            # Održavaj veličinu povijesti
            if len(self.conversation_state.dialogue_history) > self.dialogue_window:
                self.conversation_state.dialogue_history.pop(0)
            
            # Ažuriraj tip konverzacije
            self._update_conversation_type()
            
            self.conversation_state.last_speaker = speaker
    
    def reset_dialogue(self):
        """Resetira trenutni dijalog kada dođe do značajne pauze ili promjene scene."""
        self.conversation_state.current_dialogue_speakers.clear()
        self.conversation_state.conversation_type = "unknown"
        self.conversation_state.last_speaker = None
        self.conversation_state.dialogue_history.clear()
    
    def _update_conversation_type(self):
        """Određuje tip konverzacije na temelju roda govornika."""
        genders = set()
        for speaker in self.conversation_state.current_dialogue_speakers:
            gender = self.conversation_state.speaker_genders.get(speaker)
            if gender:
                genders.add(gender)
        
        if not genders:
            self.conversation_state.conversation_type = "unknown"
        elif len(genders) == 1:
            if "female" in genders:
                self.conversation_state.conversation_type = "female_only"
            elif "male" in genders:
                self.conversation_state.conversation_type = "male_only"
        else:
            self.conversation_state.conversation_type = "mixed"
    
    def get_context_for_speaker(self, speaker: str) -> Dict:
        """Dohvaća kontekst za trenutnog govornika."""
        context = {
            'conversation_type': self.conversation_state.conversation_type,
            'is_response_to': None,
            'dialogue_context': [],
            'speaker_gender': self.conversation_state.speaker_genders.get(speaker),
            'previous_speakers_gender': None
        }
        
        # Provjeri zadnjih nekoliko replika za kontekst
        recent_history = self.conversation_state.dialogue_history[-self.dialogue_window:]
        context['dialogue_context'] = recent_history
        
        # Ako je ovo odgovor na prethodnu repliku
        if recent_history and recent_history[-1][0] != speaker:
            context['is_response_to'] = recent_history[-1][0]
            context['previous_speakers_gender'] = recent_history[-1][1]
        
        return context
    
    def suggest_gender(self, speaker: str) -> Optional[str]:
        """Predlaže rod za govornika na temelju konteksta konverzacije."""
        if speaker in self.conversation_state.speaker_genders:
            return self.conversation_state.speaker_genders[speaker]
        
        # Pogledaj povijest dijaloga
        speaker_lines = [
            (s, g, t) for s, g, t in self.conversation_state.dialogue_history
            if s == speaker
        ]
        
        if not speaker_lines:
            return None
            
        # Ako govornik razgovara s nekim čiji rod znamo
        recent_interactions = self.conversation_state.dialogue_history[-self.dialogue_window:]
        for s1, g1, t1 in recent_interactions:
            if s1 != speaker and g1 != "unknown":
                # Analiziraj prirodu interakcije
                # Npr. ako dvije osobe razgovaraju romantično, vjerojatno su različitog roda
                romantic_indicators = ['love', 'darling', 'honey', 'sweetheart']
                if any(indicator in t1.lower() for indicator in romantic_indicators):
                    return 'male' if g1 == 'female' else 'female'
        
        return None